import numpy as np
import scipy.io as scio
import os
from scipy import signal
from scipy.signal import butter, filtfilt
import librosa
import matplotlib.pyplot as plt

path_Extracted = 'E:/coding/sleep_stage/data/ISRUC_S1/ExtractedChannels' #<-Change to your folder path
path_RawData = 'E:/coding/sleep_stage/data/ISRUC_S1/RawData' #<-Change to your folder path

path_output = 'ISRUC1_ECG_STFT_class4' #save directory name #ISRUC1_ECG_mel_class4
current_dir = os.getcwd()
path_output = os.path.join(current_dir, path_output)

if not os.path.exists(path_output):
    os.makedirs(path_output)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

def z_score_normalization(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    normalized_data = (data - mean) / std
    return normalized_data

def visualize_ecg(ecg_data, label, file_id):
    plt.figure(figsize=(10, 4))
    plt.plot(ecg_data, color='blue')
    plt.title(f"Raw ECG Signal: {file_id}, Label: {label_to_text[label]}")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Amplitude")

    output_dir = "./ECG_visualizations/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/{file_id}_label{label}.png")
    plt.close()


def generate_mel_spectrogram(data, fs, n_fft=256, hop_length=16, n_mels=32):
    Sxx = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                         window='hann',fmax=45)
    Sxx_db = librosa.power_to_db(Sxx, ref=np.max)


    return Sxx_db

def generate_spectrogram(data, n_fft=254, hop_length=214):
    Sxx = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, window='hann')
    Sxx_db = np.abs(Sxx)

    return Sxx_db

def visualize_mel_spectrogram(Sxx, label, file_id):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(Sxx, sr=200, hop_length=16, y_axis='mel', fmax=45, x_axis='time', cmap='magma')

    ax = plt.gca()
    ax.set_ylim(0, 45)

    plt.colorbar(format='%+2.0f dB')
    plt.title(f"File: {file_id}, Label: {label_to_text[label]}")

    output_dir = "./mel_spectrogram_visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/mel_spectrogram_{file_id}_{label_to_text[label]}.png")
    plt.close()

def visualize_spectrogram(Sxx, label, file_id):
    plt.figure(figsize=(10, 4))

    librosa.display.specshow(Sxx, sr=200, hop_length=16, y_axis='hz', fmax=45, x_axis='time')
    plt.colorbar(format='%+2.0f dB')  # dB 스케일로 색상 바 표시
    plt.title(f"File: {file_id}, Label: {label}")
    plt.ylabel("Frequency (Hz)")  # 주파수 축 레이블 수정

    output_dir = "./linear_spectrogram_visualizations_ECG/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/spectrogram_{file_id}_{label_to_text[label]}.png")
    plt.close()

def read_psg(path_Extracted, sub_id, fs=200):
     psg = scio.loadmat(os.path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
     psg_use = psg['X2'] # 'C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1', 'LOC_A2', 'ROC_A1','X1', 'X2'

     lowcut = 0.3
     highcut = 45.0
     psg_filtered = bandpass_filter(psg_use, lowcut, highcut, fs)

     psg_normalized = z_score_normalization(psg_filtered)

     return psg_normalized


def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(os.path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            lbl = int(a)

            if lbl == 1 or lbl == 2:  # N2 -> Light Sleep (same as N1)
                label.append(1)
            elif lbl == 3 or lbl == 4:  # N3, N4 -> Deep Sleep
                label.append(2)
            elif lbl == 5:  # REM
                label.append(3)
            elif lbl == 0:  # Wake (label 0 remains the same)
                label.append(0)

            # 5 class mapping
            # if lbl == 0:  # Wake
            #     label.append(0)
            # elif lbl == 1:  # N1
            #     label.append(1)
            # elif lbl == 2:  # N2
            #     label.append(2)
            # elif lbl == 3 or lbl == 4:  # N3
            #     label.append(3)
            # elif lbl == 5:  #REM
            #     label.append(4)

            s = f.readline()
            if s == '' or s == '\n':
                break

    return np.array(label[:-ignore])

# 레이블 매핑 딕셔너리
label_to_text = {
    0: 'WAKE',
    1: 'Light_Sleep',
    2: 'Deep_Sleep',
    3: 'REM'
}

# label_to_text = {
#     0: 'WAKE',
#     1: 'N1',
#     2: 'N2',
#     3: 'N3',
#     4: 'REM '
# }

for sub in range(1, 101):
    psg = read_psg(path_Extracted, sub).astype(np.float32)
    label = read_label(path_RawData, sub).astype(np.int32)

    assert len(label) == len(psg)

    spectrogram_data_list = []
    class_processed = set()
    for epoch_index, epoch in enumerate(psg):
        current_label = label[epoch_index]

        #Sxx = generate_mel_spectrogram(epoch.flatten(), fs=200)
        Sxx = generate_spectrogram(epoch.flatten())

        if current_label not in class_processed:
            visualize_spectrogram(Sxx, current_label, f'ISRUC_S1_{sub}_{current_label}')
            #visualize_mel_spectrogram(Sxx, current_label, f'ISRUC_S1_{sub}_{current_label}')
            #visualize_ecg(epoch, current_label, f'ISRUC_S1_{sub}_{current_label}')

            class_processed.add(current_label)

        spectrogram_data_list.append(Sxx)
    filename = os.path.join(path_output, 'ISRUC_S1_%d.npz' % (sub))

    spectrogram_data_array = np.array(spectrogram_data_list)
    spectrogram_file_path = os.path.join(path_output, filename)

    print('생성한 스펙트로그램 Shape : ',spectrogram_data_array.shape)
    wake_indices = np.where(label == 0)[0]

    wake_x_data = psg[wake_indices]

    print(f"Wake 데이터 개수: {len(wake_x_data)}")
    print(f"Wake 데이터 샘플:\n{wake_x_data}")

    np.savez(spectrogram_file_path, spectrogram_data=spectrogram_data_array, labels=label)
    print(f"Saved spectrogram data to {spectrogram_file_path}")

print('--preprocess over--')
    