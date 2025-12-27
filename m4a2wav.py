from pathlib import Path
from pydub import AudioSegment


def convert_m4a_to_wav(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    for m4a_path in src_dir.glob("*.m4a"):
        wav_path = dst_dir / (m4a_path.stem + ".wav")

        audio = AudioSegment.from_file(m4a_path, format="m4a")
        audio.export(wav_path, format="wav")

        print(f"Converted: {m4a_path.name} -> {wav_path.name}")


if __name__ == "__main__":
    source_path = r"C:\Users\nirfi\PycharmProjects\AudioProcessingEx2\Mom_Dad_Ran_Records_M4A"
    dest_path = r"C:\Users\nirfi\PycharmProjects\AudioProcessingEx2\Mom_Dad_Ran_Records_Wav"
    src = Path(source_path)
    dst = Path(dest_path)

    if not src.is_dir():
        raise ValueError(f"Source directory does not exist: {src}")

    convert_m4a_to_wav(src, dst)
