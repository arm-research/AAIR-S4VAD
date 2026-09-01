from pathlib import Path
from scipy.signal import resample_poly
import soundfile as sf
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset root directory')
parser.add_argument('--out_dir', type=str, required=True, help='Path to output json root directory')
parser.add_argument('--target_sr', type=float, default=16000, help='Target sample rate')


def main(
    input_root: str,
    output_root: str,
    target_sr: int = 16000,
):
    input_root = Path(input_root)
    output_root = Path(output_root)

    for wav_path in tqdm(input_root.rglob("*.wav")):
        # Relative path (to preserve folder structure)
        rel_path = wav_path.relative_to(input_root)
        out_path = output_root / rel_path

        # Create output directory if needed
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read audio
        audio, sr = sf.read(wav_path)

        # Downsample if needed
        if sr != target_sr:
            audio = resample_poly(audio, target_sr, sr)

        # Write output
        sf.write(out_path, audio, target_sr)

        print(f"✔ {wav_path} -> {out_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.dataset_dir, args.out_dir, args.target_sr)

