import os
from builtins import print

import numpy as np
import argparse
import pandas as pd
import xml.etree.ElementTree as ET
from mne.io import read_raw_edf
import librosa
import matplotlib.pyplot as plt

EPOCH_SEC_SIZE = 30

def generate_mel_spectrogram(data, fs, n_fft=256, hop_length=16, n_mels=32):
    Sxx = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                         window='hann',fmax=fs/2)
    Sxx_db = librosa.power_to_db(Sxx, ref=np.max)

    return Sxx_db

def generate_linear_spectrogram(data, n_fft=254, hop_length=260):
    Sxx = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, window='hann')
    Sxx_db = np.abs(Sxx)

    return Sxx_db

def visualize_spectrogram(Sxx, label, file_id):
    plt.figure(figsize=(10, 4))
    plt.imshow(Sxx, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label="Power (dB)")
    plt.title(f"File: {file_id}, Label: {label_to_text[label]}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.ylabel("Mel Frequency Bins")

    # Save the visualization as an image
    output_dir = "./spectrogram_visualizations_ECG/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/{file_id}_label_{label_to_text[label]}.png")
    plt.close()

def visualize_ecg(ecg_data, label, file_id):
    plt.figure(figsize=(10, 4))
    plt.plot(ecg_data, color='blue')
    plt.title(f"Raw ECG Signal: {file_id}, Label: {label_to_text[label]}")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Amplitude")

    output_dir = "./ecg_visualizations/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/{file_id}_label_{label_to_text[label]}.png")
    plt.close()

# 4-class
label_to_text = {
    0: 'WAKE',
    1: 'Light_Sleep',
    2: 'Deep_Sleep',
    3: 'REM'
}

# 5-class
# label_to_text = {
#     0: 'WAKE',
#     1: 'N1',
#     2: 'N2',
#     3: 'N3',
#     4: 'REM '
# }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="E:/coding/sleep_stage/data/shhs/polysomnography/edfs/shhs2",
                        help="File path to the PSG files.")
    parser.add_argument("--ann_dir", type=str,
                        default="E:/coding/sleep_stage/data/shhs/polysomnography/annotations-events-profusion/shhs2",
                        help="File path to the annotation files.")
    parser.add_argument("--original_data_dir", type=str, default="./shhs2_ECG_class4",
                        help="Directory to save the processed data files.")
    parser.add_argument("--spectrogram_dir", type=str, default="./shhs2_ECG_mel_class4",
                        help="Directory to save the spectrogram data files.")
    parser.add_argument("--select_ch", type=str, default="ECG",
                        help="The selected channel")
    args = parser.parse_args()


    if not os.path.exists(args.original_data_dir):
        os.makedirs(args.original_data_dir)

    if not os.path.exists(args.spectrogram_dir):
        os.makedirs(args.spectrogram_dir)

    ids = pd.read_csv("selected_shhs2_files.txt", header=None, names=['a'])
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

        sampling_rate = raw.info['sfreq']
        print(f"🔍 Original sampling rate: {sampling_rate} Hz")

        desired_sfreq = 250
        if raw.info['sfreq'] != desired_sfreq:
            print(f"🔄 Resampling from {raw.info['sfreq']} Hz to {desired_sfreq} Hz")
            raw.resample(desired_sfreq)

        sampling_rate = raw.info['sfreq']
        print(f"🔍 Changed sampling rate: {sampling_rate} Hz")

        ch_type = args.select_ch.split(" ")[0]
        select_ch = [s for s in raw.info["ch_names"] if ch_type in s][0]
        print('select_ch', select_ch)
        raw_ch_df = raw.to_data_frame(scalings=sampling_rate)[select_ch]
        raw_ch_df = raw_ch_df.to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))

        labels = []
        t = ET.parse(ann_fnames[file_id])
        r = t.getroot()
        faulty_File = 0

        # for i in range(len(r[4])): #5-class
        #     lbl = int(r[4][i].text)
        #     if lbl == 4:
        #         labels.append(3)
        #     elif lbl == 5:
        #         labels.append(4)
        #     else:
        #         labels.append(lbl)
        #     if lbl > 5:
        #         faulty_File = 1

        for i in range(len(r[4])): #0(Wake) {1(N1) 2(N2)} {3(N3) 4(N4)} {5(REM)} 4-class
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

        labels = np.asarray(labels)
        raw_ch = raw_ch_df.values
        print(raw_ch.shape)

        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Data length issue.")

        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        print(x[0])
        y = labels.astype(np.int32)

        print(x.shape)
        print(y.shape)
        assert len(x) == len(y)

        filename = os.path.basename(edf_fnames[file_id]).replace(".edf", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate
        }

        original_file_path = os.path.join(args.original_data_dir, filename)
        np.savez(original_file_path, **save_dict)
        print(f"Saved Original data to {original_file_path}")

        spectrogram_data_list = []
        class_processed = set()
        for epoch_index, epoch in enumerate(x):
            Sxx = generate_mel_spectrogram(epoch.flatten(), fs=sampling_rate)
            #Sxx = generate_linear_spectrogram(epoch.flatten())
            #visualize_spectrogram(Sxx, y[0], os.path.basename(edf_fnames[file_id]))

            # current_label = y[epoch_index] #visulization
            # if current_label not in class_processed:
            #     visualize_spectrogram(Sxx, current_label, f'SHHS2_{file_id}_{current_label}')
            #     visualize_ecg(epoch, current_label, f'SHHS2_{file_id}_{current_label}')
            #     class_processed.add(current_label)
            spectrogram_data_list.append(Sxx)

        spectrogram_data_array = np.array(spectrogram_data_list)
        spectrogram_file_path = os.path.join(args.spectrogram_dir, filename)
        np.savez(spectrogram_file_path, spectrogram_data=spectrogram_data_array, labels=y)
        print(f"Saved spectrogram data to {spectrogram_file_path}")

if __name__ == "__main__":
    main()
