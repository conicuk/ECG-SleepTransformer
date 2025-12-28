from torch.utils.data import Dataset
import torch
import numpy as np

class MelDataset(Dataset):
    def __init__(self, mel_files):
        super(MelDataset, self).__init__()

        x_mel_data = np.load(mel_files[0])["spectrogram_data"]
        y_data = np.load(mel_files[0])["labels"]

        for np_file in mel_files[1:]:
            x_mel_data = np.vstack((x_mel_data, np.load(np_file)["spectrogram_data"]))
            y_data = np.append(y_data, np.load(np_file)["labels"])

        self.x_data = torch.from_numpy(x_mel_data)
        self.y_data = torch.from_numpy(y_data).long()

        self.len = self.y_data.shape[0]
        self.x_data = self.x_data.unsqueeze(1)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class IHR_EDR(Dataset):
    def __init__(self, ihr_edr_files):
        super(IHR_EDR, self).__init__()

        x_ihr_data = np.load(ihr_edr_files[0])["ihr"]
        x_edr_data = np.load(ihr_edr_files[0])["edr"]
        y_data = np.load(ihr_edr_files[0])["y"]

        for np_file in ihr_edr_files[1:]:
            x_ihr_data = np.vstack((x_ihr_data, np.load(np_file)["ihr"]))
            x_edr_data = np.vstack((x_edr_data, np.load(np_file)["edr"]))
            y_data = np.append(y_data, np.load(np_file)["y"])

        self.x_ihr_data = torch.from_numpy(x_ihr_data)
        self.x_edr_data = torch.from_numpy(x_edr_data)
        self.y_data = torch.from_numpy(y_data).long()

        self.len = self.y_data.shape[0]
        print(f"x_ihr_data before shape: {self.x_ihr_data.shape}")
        self.x_ihr_data = self.x_ihr_data.unsqueeze(1)
        print(f"x_ihr_data after shape: {self.x_ihr_data.shape}")
        # Print shapes of the loaded data

        print(f"x_edr_data before shape: {self.x_edr_data.shape}")
        self.x_edr_data = self.x_edr_data.unsqueeze(1)
        print(f"x_edr_data after shape: {self.x_edr_data.shape}")

        print('Done')

    def __getitem__(self, index):
        return self.x_ihr_data[index], self.x_edr_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class LoadDataset_from_numpy(Dataset):
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()

        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class SleepTransformerDataset(Dataset):
    def __init__(self, npz_files, epoch_seq_len=20, nchannel=1):
        super(SleepTransformerDataset, self).__init__()

        self.epoch_seq_len = epoch_seq_len
        self.nchannel = nchannel

        X_all = []
        y_all = []

        for np_file in npz_files:
            try:
                loader = np.load(np_file)
                x_data_single = loader["spectrogram_data"]
                y_data_single = loader["labels"]

                if x_data_single.ndim != 3:
                    print(f"[WARN] {np_file} : spectrogram_data ndim={x_data_single.ndim}, not three → skip")
                    continue

                if x_data_single.shape[1] in (32, 64, 128, 256):
                    Dd = x_data_single.shape[1]
                    Tf = x_data_single.shape[2]
                else:
                    x_data_single = np.transpose(x_data_single, (0, 2, 1))
                    Dd = x_data_single.shape[1]
                    Tf = x_data_single.shape[2]

                x_data_single = np.expand_dims(x_data_single, axis=-1)
                x_data_single = np.transpose(x_data_single, (0, 2, 1, 3))

                X_all.append(x_data_single)
                y_all.append(y_data_single)

            except Exception as e:
                print(f"[ERROR] file load error: {np_file}, Error: {e}")
                continue

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        Tf_actual = X_all.shape[1]
        Dd_actual = X_all.shape[2]
        Dc_actual = X_all.shape[3]

        self.frame_seq_len = Tf_actual
        self.ndim = Dd_actual
        self.nchannel = Dc_actual

        print("=== SleepTransformerDataset ===")
        print(f"  Total epochs  : {X_all.shape[0]}")
        print(f"  Per-epoch shape (Tf, Dd, Dc): ({Tf_actual}, {Dd_actual}, {Dc_actual})")
        print(f"  epoch_seq_len (Te)          : {self.epoch_seq_len}")
        print("================================")

        n_sequences = X_all.shape[0] // self.epoch_seq_len
        used_epochs = n_sequences * self.epoch_seq_len

        X_trim = X_all[:used_epochs]
        y_trim = y_all[:used_epochs]

        X_seq = X_trim.reshape(
            n_sequences,
            self.epoch_seq_len,
            self.frame_seq_len,
            self.ndim,
            self.nchannel,
        )

        y_seq = y_trim.reshape(
            n_sequences,
            self.epoch_seq_len,
        )

        self.x_data = torch.from_numpy(X_seq).float()
        self.y_data = torch.from_numpy(y_seq).long()
        self.len = self.x_data.shape[0]

        print("--- SleepTransformer Dataset Summary ---")
        print(f"  Total Sequences      : {self.len}")
        print(f"  X shape (N_seq, Te, Tf, Dd, Dc): {self.x_data.shape}")
        print(f"  Y shape (N_seq, Te)            : {self.y_data.shape}")
        print("----------------------------------------")

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def data_generator_np(total_data, data_type):
    if data_type == "ECG_IHR":
        dataset = LoadDataset_from_numpy(total_data) #ecg signal
    elif data_type == "Mel":
        dataset = MelDataset(total_data)  # mel-spectrogram

    test_x, test_y = dataset.x_data.numpy(), dataset.y_data.numpy()

    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y).long()

    print(f"ECG data shape: {len(test_x)}, Labels shape: {len(test_y)}")

    unique_train_classes, train_counts = torch.unique(test_y, return_counts=True)

    print("Train class counts:")
    for cls, count in zip(unique_train_classes, train_counts):
        print(f"Class {cls.item()}: {count.item()} samples")

    return test_x, test_y

def data_generator_np_IHR_EDR(total_data):
    dataset = IHR_EDR(total_data)

    x_ihr, x_edr, y = dataset.x_ihr_data.numpy(), dataset.x_edr_data.numpy(), dataset.y_data.numpy()

    x_ihr = torch.from_numpy(x_ihr)
    x_edr = torch.from_numpy(x_edr)
    y = torch.from_numpy(y).long()

    print(f"IHR data shape: {len(x_ihr)}, EDR data shape: {len(x_edr)}, Labels shape: {len(y)}")

    unique_train_classes, train_counts = torch.unique(y, return_counts=True)

    print("Train class counts:")
    for cls, count in zip(unique_train_classes, train_counts):
        print(f"Class {cls.item()}: {count.item()} samples")

    return x_ihr, x_edr, y

def data_generator_sleeptransformer(npz_files, config_params):
    Te = config_params.get("epoch_seq_len", 20)
    Dc = config_params.get("nchannel", 1)

    dataset = SleepTransformerDataset(
        npz_files,
        epoch_seq_len=Te,
        nchannel=Dc,
    )

    X_seq = dataset.x_data
    y_seq = dataset.y_data

    total_y = y_seq.flatten()
    unique_classes, counts = torch.unique(total_y, return_counts=True)

    print("\n[SleepTransformer] Total Epoch Class counts in Dataset:")
    for cls, count in zip(unique_classes, counts):
        print(f"  Class {cls.item()}: {count.item()} samples")

    return X_seq, y_seq