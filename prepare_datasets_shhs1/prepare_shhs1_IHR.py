import os
from builtins import print

import numpy as np
import argparse
import pandas as pd
import xml.etree.ElementTree as ET
from mne.io import read_raw_edf
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d

###############################
EPOCH_SEC_SIZE = 30
RESAMPLE_FS = 2.0    # Hz for IHR/EDR signals
WINDOW_SEC = 128  # seconds for window length

#### 전처리 함수 추가 ####
def preprocess_signal(data, fs, lowcut=5.0, highcut=15.0, order=2):
        b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
        filtered_data = filtfilt(b, a, data, axis=-1)
        return filtered_data

def extract_ihr(ecg, fs, resample_fs=2.0):
    ecg_norm = preprocess_signal(ecg, fs, 5, 15)

    min_dist = int(0.25 * fs)
    peaks, _ = find_peaks(ecg_norm, distance=min_dist, height=np.mean(ecg_norm))

    ibi = np.diff(peaks) / fs

    ibi_mean, ibi_std = np.mean(ibi), np.std(ibi)
    mask = np.abs(ibi - ibi_mean) <= 5 * ibi_std

    ibi_clean = ibi[mask]
    rr_times = peaks[1:][mask] / fs

    if len(ibi_clean) < 2:
        L = int(len(ecg) / fs * resample_fs)
        return np.zeros(L, dtype=np.float32)

    ihr = 1.0 / ibi_clean
    ihr_z = (ihr - ihr.mean()) / (ihr.std() + 1e-8)

    duration = len(ecg) / fs
    t_new = np.arange(0, duration, 1.0 / resample_fs)
    f = interp1d(rr_times, ihr_z, bounds_error=False, fill_value='extrapolate')
    return f(t_new).astype(np.float32)

def visualize_ihr(ecg, ihr, fs, out_dir, basename, epoch_idx=0):
    samples_ecg = int(EPOCH_SEC_SIZE * fs)
    samples_res = int(EPOCH_SEC_SIZE * RESAMPLE_FS)

    start_ecg = epoch_idx * samples_ecg
    ecg_seg = ecg[start_ecg:start_ecg + samples_ecg]
    t_ecg = np.arange(len(ecg_seg)) / fs

    start_res = epoch_idx * samples_res
    ihr_seg = ihr[start_res:start_res + samples_res]
    t_res = np.arange(len(ihr_seg)) / RESAMPLE_FS

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 3))
    plt.plot(t_ecg, ecg_seg, color='black')
    plt.title(f'ECG (Epoch {epoch_idx}): {basename}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{basename}_epoch{epoch_idx}_ECG.png'))
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(t_res, ihr_seg, color='red')
    plt.title(f'IHR (2Hz, Epoch {epoch_idx}): {basename}')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized HR')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{basename}_epoch{epoch_idx}_IHR.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="E:/coding/sleep_stage/data/shhs/polysomnography/edfs/shhs1", #<- Change to your folder path
                        help="File path to the PSG files.")
    parser.add_argument("--ann_dir", type=str,
                        default="E:/coding/sleep_stage/data/shhs/polysomnography/annotations-events-profusion/shhs1", #<- Change to your folder path
                        help="File path to the annotation files.")
    parser.add_argument("--original_data_dir", type=str, default="./shhs1_IHR_class4", #default: save directory name
                        help="Directory to save the processed data files.")
    parser.add_argument("--select_ch", type=str, default="ECG",
                        help="The selected channel")
    args = parser.parse_args()

    os.makedirs(args.original_data_dir, exist_ok=True)

    ids = pd.read_csv("selected_shhs1_files.txt", header=None, names=['a'])
    ids = ids['a'].values.tolist()

    edf_fnames = [os.path.join(args.data_dir, i + ".edf") for i in ids]
    ann_fnames = [os.path.join(args.ann_dir, i + "-profusion.xml") for i in ids]

    edf_fnames.sort()
    ann_fnames.sort()

    for file_id in range(len(edf_fnames)):
        if os.path.exists(os.path.join(args.original_data_dir, edf_fnames[file_id].split('/')[-1])[:-4] + ".npz"):
            continue
        print(f"Processing file: {edf_fnames[file_id]}")

        raw = read_raw_edf(edf_fnames[file_id], preload=True, stim_channel=None, verbose=None)
        fs = raw.info['sfreq']
        print(f"🔍 Original sampling rate: {fs} Hz")

        ch = [c for c in raw.ch_names if args.select_ch in c][0]
        raw.pick([ch])
        ecg = raw.get_data().flatten()
        print('select_ch', ch)

        labels = []
        t = ET.parse(ann_fnames[file_id])
        r = t.getroot()
        faulty_File = 0

        for i in range(len(r[4])): #0(Wake) {1(N1) 2(N2)} {3(N3) 4(N4)} {5(REM)}
            lbl = int(r[4][i].text)
            if lbl == 2:
                labels.append(1)
            elif lbl == 3 or lbl == 4:
                labels.append(2)
            elif lbl == 5:
                labels.append(3)
            else:
                labels.append(lbl)
            if lbl > 5:
                faulty_File = 1

        if faulty_File == 1:
            print(f"⚠️ Faulty label in file {edf_fnames[file_id]} — skipped.")
            continue

        y = np.array(labels, dtype=np.int32)
        n_epochs = len(y)

        filename = os.path.basename(edf_fnames[file_id]).replace(".edf", ".npz")

        ihr_full = extract_ihr(ecg, fs, RESAMPLE_FS)

        win_len = int(WINDOW_SEC * RESAMPLE_FS)
        half_w = win_len // 2
        ihr_pad = np.pad(ihr_full, (half_w, half_w), mode='edge')

        ihr_epochs = np.zeros((n_epochs, win_len), dtype=np.float32)

        for i in range(n_epochs):
            center = int((i + 0.5) * EPOCH_SEC_SIZE * RESAMPLE_FS)
            start = center
            ihr_epochs[i] = ihr_pad[start:start + win_len]

        # vis_dir = os.path.join(args.original_data_dir, "visualizations")
        # basename = os.path.basename(edf_fnames[file_id]).replace(".edf", "")
        #
        # visualize_ihr(ecg, ihr_full, fs, vis_dir, basename, epoch_idx=1)
        # print(f"🔍 Saved epoch-1 visualization to {vis_dir}/{basename}_epoch1_ihr.png")

        IHR_path = os.path.join(args.original_data_dir, filename)

        np.savez(IHR_path, x=ihr_epochs, y=y)
        print(f"Saved data (IHR, EDR, labels) to {IHR_path}: Label Shape={y.shape}, IHR shape={ihr_epochs.shape}")

if __name__ == "__main__":
    main()
