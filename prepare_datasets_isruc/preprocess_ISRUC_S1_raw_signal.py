import numpy as np
import scipy.io as scio
import os
from scipy import signal
from scipy.signal import butter, filtfilt

path_Extracted = 'E:/coding/sleep_stage/data/ISRUC_S1/ExtractedChannels' #<-Change to your folder path
path_RawData = 'E:/coding/sleep_stage/data/ISRUC_S1/RawData' #<-Change to your folder path

path_output = 'ISRUC1_ECG_class4' #save directory name
current_dir = os.getcwd()
path_output = os.path.join(current_dir, path_output)

if not os.path.exists(path_output):
    os.makedirs(path_output)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    # nyquist = 0.5 * fs
    # low = lowcut / nyquist
    # high = highcut / nyquist
    # b, a = butter(order, [low, high], btype='band')
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

def z_score_normalization(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    normalized_data = (data - mean) / std
    return normalized_data

def read_psg(path_Extracted, sub_id, fs=200):
     psg = scio.loadmat(os.path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
     psg_use = psg['X2'] # 'C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1', 'LOC_A2', 'ROC_A1','X1', 'X2'
     print('psg_use',psg_use)
     print('psg.shape:',psg_use.shape)
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

            # 4 class mapping
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
            # elif lbl == 5:  # REM
            #     label.append(4)

            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])

for sub in range(1, 101):
    psg = read_psg(path_Extracted, sub).astype(np.float32)
    label = read_label(path_RawData, sub).astype(np.int32)

    assert len(label) == len(psg)

    filename = os.path.join(path_output, 'ISRUC_S1_%d.npz' % (sub))

    save_dict = {'x': psg, 'y': label}

    wake_indices = np.where(label == 0)[0]

    wake_x_data = psg[wake_indices]

    np.savez(filename, **save_dict)
    print(f"Saved spectrogram data to {filename}")
print('--preprocess over--')
    