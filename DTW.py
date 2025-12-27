import time
from typing import Dict, List
from pathlib import Path
from scipy.spatial.distance import cdist
import librosa
import numpy as np
from numpy.typing import NDArray
from numba import njit

RANDOM_RECORD_NAME = "random"
RECORD_NAMES = [str(i) for i in range(10)]
RECORD_NAMES_WITH_RANDOM = RECORD_NAMES + [RANDOM_RECORD_NAME]

WINDOW_SIZE = 25
HOP_SIZE = 10
FILTER_BANKS = 80


class PersonRecording:
    def __init__(self, recording: NDArray[np.float32], mel_spectrom, sample_rate: int, recorder_name: str,
                 recording_name: str):
        self.raw_recording = recording
        self.mel_spectrom = mel_spectrom
        self.sample_rate = sample_rate
        self.recorder_name = recorder_name
        self.record_name = recording_name


class PersonRecordings:
    def __init__(self, recorder_name):
        self.recoder_name = recorder_name
        self.recordings: Dict[str, PersonRecording] = {}

    def add_recording(self, recording: PersonRecording):
        self.recordings[recording.record_name] = recording

    def get_recording(self, number):
        return self.recordings[number]


class RecordingDataset:
    def __init__(self, reference: PersonRecordings, train: List[PersonRecordings], validation: List[PersonRecordings]):
        self.reference = reference
        self.training_set = train
        self.validation_set = validation
        self.dtw = np.full((4, 10, 11), -1)

    def train(self):
        for i, person in enumerate(self.training_set):
            for j, train_record_name in enumerate(RECORD_NAMES):
                current_train_record = self.training_set[i].get_recording(train_record_name)
                for k, reference_record_name in enumerate(RECORD_NAMES_WITH_RANDOM):
                    current_reference_record = self.reference.get_recording(reference_record_name)
                    s = time.time()
                    self.dtw[i][j][k] = calc_dtw(current_train_record.mel_spectrom.T,
                                                 current_reference_record.mel_spectrom.T)
                    print('took: ', time.time() - s)


def load_dataset(path: str) -> RecordingDataset:
    root = Path(path)

    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    reference_dir = root / "reference"
    train_dir = root / "train"
    validation_dir = root / "validation"

    for d in (reference_dir, train_dir, validation_dir):
        if not d.is_dir():
            raise ValueError(f"Missing expected directory: {d}")

    # -------- Reference (iterate over recorder dirs) --------
    reference_dirs = [
        d for d in reference_dir.iterdir() if d.is_dir()
    ]

    if len(reference_dirs) != 1:
        raise ValueError(
            f"Reference directory must contain exactly one recorder directory, "
            f"found {len(reference_dirs)}"
        )

    reference = load_person_recordings(
        path=str(reference_dirs[0]),
        is_reference=True,
    )

    if not reference:
        raise ValueError("Reference directory must contain at least one recorder directory")

    # -------- Train --------
    train: List[PersonRecordings] = []
    for person_dir in train_dir.iterdir():
        if person_dir.is_dir():
            train.append(
                load_person_recordings(
                    path=str(person_dir),
                    is_reference=False,
                )
            )

    # -------- Validation --------
    validation: List[PersonRecordings] = []
    for person_dir in validation_dir.iterdir():
        if person_dir.is_dir():
            validation.append(
                load_person_recordings(
                    path=str(person_dir),
                    is_reference=False,
                )
            )

    return RecordingDataset(
        reference=reference,
        train=train,
        validation=validation,
    )


def load_person_recordings(path: str, is_reference: bool) -> PersonRecordings:
    person_dir = Path(path)

    if not person_dir.is_dir():
        raise ValueError(f"Not a directory: {person_dir}")

    name = person_dir.name
    person_recordings = PersonRecordings(recorder_name=name)

    random_found = False

    for wav_path in person_dir.glob("*.wav"):
        stem = wav_path.stem  # filename without .wav

        # ---- Special case: reference random file ----
        if is_reference and stem == f"{name} random":
            recording, sample_rate = librosa.load(
                wav_path,
                sr=16000,
                mono=True,
            )

            raw_recording = np.asarray(recording, dtype=np.float32)
            mel_power_spectrogram = librosa.feature.melspectrogram(
                y=raw_recording,
                sr=sample_rate,
                n_fft=FILTER_BANKS,
                hop_length=HOP_SIZE,
                win_length=WINDOW_SIZE
            )

            person_recordings.add_recording(
                PersonRecording(
                    recording=np.asarray(recording, dtype=np.float32),
                    mel_spectrom=mel_power_spectrogram,
                    sample_rate=sample_rate,
                    recorder_name=name,
                    recording_name="random",
                )
            )

            random_found = True
            continue

        # ---- Normal numbered recordings ----
        try:
            _, digit_str = stem.rsplit(" ", 1)
        except ValueError:
            raise ValueError(f"Invalid filename format: {wav_path.name}")

        if not (len(digit_str) == 1 and digit_str.isdigit()):
            raise ValueError(f"Expected single digit number in filename: {wav_path.name}")

        recording, sample_rate = librosa.load(
            wav_path,
            sr=16000,
            mono=True,
        )

        raw_recording = np.asarray(recording, dtype=np.float32)
        mel_power_spectrogram = librosa.feature.melspectrogram(
            y=raw_recording,
            sr=sample_rate,
            n_mels=FILTER_BANKS,
            hop_length=HOP_SIZE,
            win_length=WINDOW_SIZE
        )

        person_recordings.add_recording(
            PersonRecording(
                recording=np.asarray(recording, dtype=np.float32),
                mel_spectrom=mel_power_spectrogram,
                sample_rate=sample_rate,
                recorder_name=name,
                recording_name=digit_str,
            )
        )

    # ---- Validation: random must exist for reference ----
    if is_reference and not random_found:
        raise ValueError(
            f"Reference person '{name}' must contain '{name} random.wav'"
        )

    return person_recordings


def cell_value_or_inf(accumulated_cost_matrix, index_in_record_b, index_in_record_a):
    if index_in_record_a < 0 or index_in_record_b < 0:
        return np.inf
    return accumulated_cost_matrix[index_in_record_b][index_in_record_a]


def update_cost_in_current_indices(accumulated_cost_matrix: NDArray[np.float32],
                                   dist_matrix,
                                   index_record_b: int,
                                   index_record_a: int):
    current_cell_value = dist_matrix[index_record_b, index_record_a]
    accumulated_cost_matrix[index_record_b, index_record_a] = current_cell_value + \
                                                              min(cell_value_or_inf(accumulated_cost_matrix,
                                                                                    index_record_b - 1, index_record_a),
                                                                  cell_value_or_inf(accumulated_cost_matrix,
                                                                                    index_record_b, index_record_a - 1),
                                                                  cell_value_or_inf(accumulated_cost_matrix,
                                                                                    index_record_b - 1,
                                                                                    index_record_a - 1))


def calc_dtw(record_a: NDArray[np.float32], record_b: [np.float32]):
    dist_matrix = cdist(record_b, record_a, metric='sqeuclidean')
    return calc_dtw_c(record_a, record_b, dist_matrix)


@njit
def calc_dtw_c(record_a: NDArray[np.float32], record_b: [np.float32], dist_matrix):
    accumulated_cost_matrix = np.full((len(record_b) + 1, len(record_a) + 1), np.inf)
    accumulated_cost_matrix[0][0] = 0
    for index_record_b in range(1, len(record_b) + 1):
        for index_record_a in range(1, len(record_a) + 1):
            accumulated_cost_matrix[index_record_b, index_record_a] = dist_matrix[index_record_b - 1, index_record_a - 1] + \
                                                          min(accumulated_cost_matrix[index_record_b - 1, index_record_a],
                                                              accumulated_cost_matrix[index_record_b, index_record_a - 1],
                                                              accumulated_cost_matrix[index_record_b - 1, index_record_a - 1])
            # update_cost_in_current_indices(accumulated_cost_matrix,
            #                                dist_matrix,
            #                                index_record_b,
            #                                index_record_a)
    return accumulated_cost_matrix[len(record_b), len(record_a)]


def main():
    dataset_path = "records"
    dataset = load_dataset(dataset_path)
    dataset.train()
    x = 1


if __name__ == '__main__':
    main()
