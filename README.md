# ECG-SleepTransformer
Official implementation of "ECG-based Sleep Stage Classification using a CNN–Transformer with Mel-Spectrogram Representation".

Latest Update: 2025/12/31

PyTorch code for preprocessing, training, and evaluation, including:
- 🎛️ Mel-spectrogram preprocessing
- 🧠 Hybrid **CNN-Transformer** Encoder (Proposed)
- 📌 Multiple baselines (including SleepTransformer baseline4)

> ✅ This repository is intended to facilitate reproducibility.  
> ⚠️ Datasets are **NOT** redistributed here. Please download them from the official sources.

---

## 📁 Project Structure

```text
.
├── config/
├── data/
│   ├── ISRUC_S1/
│   ├── mesa/
│   └── shhs/polysomnography/
├── data_loader/
├── models/
├── prepare_datasets_isruc/
├── prepare_datasets_mesa/
├── prepare_datasets_shhs1/
├── prepare_datasets_shhs2/
├── train_script/
├── utils/
├── LICENSE
└── README.md
```

### ✅ Data folder convention
Each dataset folder in `data/` contains a `download_address.txt` file with the **official dataset URL**.
Download the dataset and place it under the corresponding folder.

---

## 1️⃣ Download Datasets

Go to:
- `data/ISRUC_S1/download_address.txt`
- `data/mesa/download_address.txt`
- `data/shhs/polysomnography/download_address.txt`

and download datasets from the official links into the matching folder.

---

## 2️⃣ Preprocessing (Generate Raw / Spectrogram / IHR / IHR+EDR)

Preprocessing scripts are provided under `prepare_datasets_*`.

### Example: ISRUC-1 scripts
- `prepare_datasets_isruc/preprocess_ISRUC_S1_raw_signal.py`
- `prepare_datasets_isruc/preprocess_ISRUC_S1_spectrogram.py`
- `prepare_datasets_isruc/preprocess_ISRUC_S1_IHR.py`
- `prepare_datasets_isruc/preprocess_ISRUC_S1_IHR_EDR.py`

### ⚠️ Important: Update local paths inside each script
Inside each preprocessing script, find:
`# <- Change to your folder path`
and replace it with **your local directory path**.

Run (example):

```bash
python prepare_datasets_isruc/preprocess_ISRUC_S1_spectrogram.py
```

After preprocessing, each script will create a **processed dataset directory** (output folder name depends on the script).

---

## 3️⃣ Configure Experiment (YAML)

Edit a YAML file in `config/` (e.g., `config/sleep_stage.yaml`).

### Key fields
- `task`: dataset name
  - `isruc1`, `shhs1`, `shhs2`, `mesa`
- `data`: input type
  - `Mel`, `ECG_IHR`, `IHR_EDR`, etc.
- `data_files`: path to your processed dataset directory
  - ⚠️ If this is an absolute Windows path (e.g., `E:\...`), change it to your own path.
- Training hyperparameters: `batch_size`, `lr`, `max_epochs`, `k_folds`, `patience`, etc.

---

## 4️⃣ Training & Evaluation

### ✅ A) Proposed + Baseline1/2/3 (main training script)

Run:

```bash
python train_script/train_sleep.py config/sleep_stage.yaml
```

In `train_script/train_sleep.py`, choose the model around:

```python
model = Proposed()  # Baseline1() / Baseline2() / Baseline3() / Proposed()
```

- `Proposed()` : ✅ Mel-spectrogram + CNN-Transformer (proposed)
- `Baseline1()` : Raw ECG baseline
- `Baseline2()` : IHR + EDR baseline (**two-input model**)
- `Baseline3()` : IHR baseline

---

## 🚨 Special Note: Baseline2 requires two-input dataloader loops (Train/Val/Test)

If you use **Baseline2**, you MUST switch the loop format in `train_script/train_sleep.py`.

### ✅ Training loop

Default (single input) — comment this out:

```python
for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)
    outputs = model(x)
```

Enable Baseline2 (two inputs) — uncomment this:

```python
for x, x2, y in train_loader:  # If you want use Baseline2
    x = x.to(device)
    x2 = x2.to(device)
    y = y.to(device)
    outputs = model(x, x2)
```

### ✅ Validation loop

Do the same switching for validation:

```python
for x, x2, y in val_loader:  # If you want use Baseline2
    ...
    outputs = model(x, x2)
```

### ✅ Testing loop (IMPORTANT)

Baseline2 ALSO requires two inputs during testing.

Default (single input) — comment this out:

```python
for x, y in test_loader:
    x = x.to(device)
    y = y.to(device)
    pred = model(x)
```

Enable Baseline2 testing — uncomment this:

```python
for x, x2, y in test_loader:  # If you want use Baseline2
    x = x.to(device)
    x2 = x2.to(device)
    y = y.to(device)
    pred = model(x, x2)
```

✅ Summary: For Baseline2, you must switch the loop format in **train/val/test**.

---

## ✅ B) Baseline4 (SleepTransformer) — separate script

Baseline4 uses a different input/shape handling and training pipeline.

Run:

```bash
python train_script/train_sleep_baseline4.py config/sleep_stage.yaml
```

> 💡 Tip: Baseline4 may require additional config fields such as `epoch_seq_len`, `frame_seq_len`, `ndim`, `nchannel`.

---

## 📦 Outputs

Training outputs are saved under `runs/` (timestamped), including:
- checkpoints
- predictions
- evaluation metrics (confusion matrix / classification report, etc.)

---

## 🧯 Troubleshooting

### ❓ "File path not found"
- Ensure you updated paths in preprocessing scripts (`# <- Change to your folder path`)
- Ensure `data_files:` in YAML points to your processed dataset folder

### ❓ Baseline2 shape/runtime errors
- You likely forgot to switch the train/val/test loops to the two-input version.

---

## 🧾 License

See `LICENSE`.

---

## 📌 Citation

If you use this codebase, please cite:

**End-to-End Sleep Stage Classification from Single-Channel ECG via Mel-Spectrogram and Hybrid CNN-Transformer Encoder**  
(Manuscript under review)

> (Optional) After publication, you can add a `CITATION.cff` file so GitHub displays citation info automatically.
