import os
import random
import soundfile as sf
from tqdm import tqdm
import numpy as np
import json
import argparse

target_sr = 16000

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset root directory')
parser.add_argument('--out_dir', type=str, required=True, help='Path to output json root directory')
parser.add_argument('--val_ratio', type=float, default=0.1, help='validation size')
parser.add_argument('--test_ratio', type=float, default=0.1, help='test size')
np.random.seed(21)


def main(args):
    dataset_dir = args.dataset_dir
    train_file, val_file, test_file = [], [], []

    # Calc number of files
    n_files = 0
    for root, dirs, files in os.walk(dataset_dir):
        n_files += len(files)

    n_val = int(args.val_ratio*n_files)
    n_test = int(args.test_ratio*n_files)

    all_indices = list(range(n_files))
    val_index = set(random.sample(all_indices, n_val))
    remaining = list(set(all_indices) - val_index)
    test_index = set(random.sample(remaining, n_test))

    with os.scandir(dataset_dir) as it:
        i = 0
        for entry in tqdm(it):
            path = os.path.join(dataset_dir, entry)

            if os.path.isdir(path):
                files_path = [os.path.join(path,x) for x in  os.listdir(path)]
            else:
                files_path = [path]

            for file_path in files_path:
                if not file_path.endswith(".wav"):
                    continue

                file = sf.SoundFile(file_path)
                if i in val_index:
                    val_file.append({'path': file_path, 'duration': len(file), 'sample_rate': file.samplerate,
                                     '16kHz_duration': int(len(file) / file.samplerate * target_sr)})
                elif i in test_index:
                    test_file.append({'path': file_path, 'duration': len(file), 'sample_rate': file.samplerate,
                                     '16kHz_duration': int(len(file) / file.samplerate * target_sr)})
                else:
                    train_file.append({'path': file_path, 'duration': len(file), 'sample_rate': file.samplerate,
                                       '16kHz_duration': int(len(file) / file.samplerate * target_sr)})
                i += 1

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'train.json'), 'w') as outfile:
        json.dump(train_file, outfile, indent=4)
    with open(os.path.join(args.out_dir, 'val.json'), 'w') as outfile:
        json.dump(val_file, outfile, indent=4)
    with open(os.path.join(args.out_dir, 'test.json'), 'w') as outfile:
        json.dump(test_file, outfile, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
