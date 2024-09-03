from reverse_detection import *
from READ_MAT import *
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import re
from scipy.signal import find_peaks



def process_file(file_path, idx):
    x = []  # Raw signal beat segments
    y = []  # Reverse label
    z = []  # Patient ID
    reverse = 0
    id = re.sub(".mat", "", os.path.basename(file_path))

    try:
        # Load .mat file
        file = load_mat(file_path)
        data = read_ecg(file)
        data = data[file['data_lost'] == False]
        sr = read_sr(file)
        r_peaks, mean_interval = read_rpeaks(file)
        beat_length = int(mean_interval * 0.7)
        beat_num = 1000
        split_3 = len(r_peaks) // 5
        ensemble = []

        for split_idx in range(0, len(r_peaks), split_3):
            if len(r_peaks) < split_idx + beat_num:
                continue
            for temp_idx in range(beat_num):
                rpeak = int(r_peaks[split_idx + temp_idx])
                segment = data[rpeak - int(beat_length * 0.4):rpeak + int(beat_length * 0.6)]
                filtered_segment = median_filter(segment)
                ensemble.append(filtered_segment)

        if len(ensemble) <= beat_num:
            return None

        arr = np.array(ensemble)
        for i in range(beat_num, len(ensemble) + 1, beat_num):
            avg = np.average(arr[:i], axis=0)
            x.append(avg)
            y.append(reverse)
            z.append(id)

    except Exception as e:
        pass
        print(e)

    return x, y, z

if __name__ == "__main__":
    path = "/workspace/data/"
    mat_data = glob.glob(os.path.join(path, "**", "*.mat"), recursive=True)[:5]
    X = []  # Raw signal data
    Y = []  # Reverse labels
    Z = []  # Patient IDs
    max_len = 0

    # 첫 번째 패스: 신호 세그먼트의 최대 길이 결정
    for idx, file_path in tqdm(enumerate(mat_data), total=len(mat_data)):
        try:
            x, _, _ = process_file(file_path, idx)
            if x:
                max_len = max(max_len, len(x[0]))
        except Exception as e:
            print(e)
            pass

    # 두 번째 패스: 파일 처리 및 시퀀스 패딩
    for idx, file_path in tqdm(enumerate(mat_data), total=len(mat_data)):
        try:
            x, y, z = process_file(file_path, idx)
            if z:
                x_padded = [np.pad(segment, (0, max_len - len(segment)), 'constant') for segment in x]
                X.extend(x_padded)
                Y.extend(y)
                Z.extend(z)
        except Exception as e:
            print(e)
            pass
    
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    np.savez(f"/workspace/output/test_with_ids.npz", x=X, y=Y, z=Z)
