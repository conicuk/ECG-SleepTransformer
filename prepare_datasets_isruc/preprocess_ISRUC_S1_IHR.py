import numpy as np
import scipy.io as scio
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d

path_Extracted = 'E:/coding/sleep_stage/data/ISRUC_S1/ExtractedChannels' #<- Change to your folder path
path_RawData = 'E:/coding/sleep_stage/data/ISRUC_S1/RawData' #<- Change to your folder path

EPOCH_SEC_SIZE = 30          # per epoch length
RESAMPLE_FS    = 2.0         # IHR/EDR resampling frequency (Hz)
PATCH_DURATION_SEC = 128        # 2 minute window

path_output = 'ISRUC1_IHR_class4' #save directory name
current_dir = os.getcwd()
path_output = os.path.join(current_dir, path_output)

if not os.path.exists(path_output):
    os.makedirs(path_output)

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

def extract_ihr(ecg, fs, resample_fs=2.0):
    ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)

    ecg_norm = bandpass_filter(ecg_norm, lowcut=5.0, highcut=15.0, fs=fs)

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

def read_psg(path_Extracted, sub_id, fs=200):
     psg = scio.loadmat(os.path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
     psg_use = psg['X2'] # 'C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1', 'LOC_A2', 'ROC_A1','X1', 'X2'
     return psg_use.astype(np.float32)

def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(os.path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            lbl = int(a)

            # 4 class mapping
            if lbl == 1 or lbl == 2:
                label.append(1)
            elif lbl == 3 or lbl == 4:
                label.append(2)
            elif lbl == 5:
                label.append(3)
            elif lbl == 0:
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
    fs = 200
    psg = read_psg(path_Extracted, sub).astype(np.float32)
    label = read_label(path_RawData, sub).astype(np.int32)
    n_epochs = len(label)

    assert len(label) == len(psg)

    psg = psg.flatten()

    ihr_full = extract_ihr(psg, fs, RESAMPLE_FS)

    win_len = int(PATCH_DURATION_SEC * RESAMPLE_FS)
    half_w = win_len // 2
    ihr_pad = np.pad(ihr_full, (half_w, half_w), mode='edge')

    ihr_epochs = np.zeros((n_epochs, win_len), dtype=np.float32)

    half_patch_sec = PATCH_DURATION_SEC // 2
    half_patch_samples = int(half_patch_sec * RESAMPLE_FS)

    for i in range(n_epochs):
        center = int((i + 0.5) * EPOCH_SEC_SIZE * RESAMPLE_FS)
        start = center
        ihr_epochs[i] = ihr_pad[start:start + win_len]

    # vis_dir = os.path.join(path_output, 'visualizations')
    # basename = f'ISRUC1_sub{sub:03d}'
    #
    # visualize_ihr(psg, ihr_full, fs, vis_dir, basename, epoch_idx=1)
    # print(f"🔍 Saved epoch-1 visualization to {vis_dir}/{basename}_epoch1_ihr.png")

    filename = os.path.join(path_output, 'ISRUC_S1_%d.npz' % (sub))

    wake_indices = np.where(label == 0)[0]


    wake_x_data = psg[wake_indices]

    print(f"Wake 데이터 개수: {len(wake_x_data)}")
    print(f"Wake 데이터 샘플:\n{wake_x_data}")

    np.savez(filename, x=ihr_epochs, y=label)
    print(f"[Subject {sub:03d}] saved: epochs={n_epochs}, IHR shape={ihr_epochs.shape}")
print('--preprocess over--')
    