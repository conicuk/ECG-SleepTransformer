import os
import sys
import time
import yaml
import random
import gc
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import *
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from data_loader.data_loaders import *
from utils.visualization_utils import save_predictions_and_evaluate, save_predictions_and_evaluate_fold1
from models.Baseline import SleepTransformer, Config
from utils.util import *

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    gc.collect()
    torch.cuda.empty_cache()

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
    else:
        params_filename = '../config/sleep_stage.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True


    if params['task'] == "shhs1" or params['task'] == "shhs2":
        total_data = load_data_shhs(params['data_files'], params['task'])

    elif params['task'] == "isruc1":
        total_data = load_data_isruc_s1(params['data_files'])

    elif params['task'] == "mesa":
        total_data = load_data_mesa(params['data_files'])
    else:
        raise ValueError(f"Unsupported task: {params['task']}")

    st_config = {
        'epoch_seq_len': params.get('epoch_seq_len', 20),
        'frame_seq_len': params.get('frame_seq_len', 29),
        'ndim': params.get('ndim', 128),
        'nchannel': params.get('nchannel', 1),
    }

    data_x, data_y = data_generator_sleeptransformer(total_data, st_config)
    dataset = TensorDataset(data_x, data_y)

    num_classes = len(torch.unique(data_y))
    print(f"Dataset Created for SleepTransformer. #Classes = {num_classes}")
    print(f"data_x shape: {tuple(data_x.shape)}, data_y shape: {tuple(data_y.shape)}")

    kfolds = params['k_folds']
    kfold = KFold(n_splits=kfolds, shuffle=True, random_state=params['random_seed'])

    criterion = nn.CrossEntropyLoss()

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    fold_accs = {'val': [], 'test': []}
    total_predictions, total_labels = [], []

    print('========================================')
    print("Start training...")

    class_names = ["W", "Light", "Deep", "REM"] if num_classes == 4 else ["W", "N1", "N2", "N3", "REM"]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold + 1}/{kfolds} ---")

        train_subset_idx, val_subset_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_subset_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_subset_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], sampler=train_subsampler, num_workers=4)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], sampler=val_subsampler, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], sampler=test_subsampler, num_workers=4)

        model_config = Config()
        model_config.epoch_seq_len = st_config['epoch_seq_len']
        model_config.frame_seq_len = st_config['frame_seq_len']
        model_config.ndim = st_config['ndim']
        model_config.nchannel = st_config['nchannel']
        model_config.nclass = num_classes

        model = SleepTransformer(model_config).to(device)

        lr = params['optimizer_params'][params['optimizer']]['lr']
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_acc = 0.0
        counter = 0

        for epoch in range(params['max_epochs']):
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Fold {fold+1} EP {epoch+1}", leave=False)
            for (x, y) in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad()

                scores, predictions = model(x)

                scores_flat = scores.reshape(-1, num_classes)
                y_flat = y.reshape(-1)

                loss = criterion(scores_flat, y_flat)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()

                preds_flat = predictions.reshape(-1)
                train_correct += (preds_flat == y_flat).sum().item()
                train_total += y_flat.size(0)

            train_acc = 100.0 * train_correct / max(1, train_total)
            train_ave_loss = train_loss_sum / max(1, len(train_loader))

            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.inference_mode():
                for (x, y) in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    scores, predictions = model(x)
                    scores_flat = scores.reshape(-1, num_classes)
                    y_flat = y.reshape(-1)

                    loss = criterion(scores_flat, y_flat)
                    val_loss_sum += loss.item()

                    preds_flat = predictions.reshape(-1)
                    val_correct += (preds_flat == y_flat).sum().item()
                    val_total += y_flat.size(0)

            val_acc = 100.0 * val_correct / max(1, val_total)
            val_ave_loss = val_loss_sum / max(1, len(val_loader))

            print(f"EP {epoch+1}: Train Loss {train_ave_loss:.4f} Acc {train_acc:.2f}% | "
                  f"Val Loss {val_ave_loss:.4f} Acc {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model_fold_{fold+1}.pth"))
            else:
                counter += 1
                if counter >= params['patience']:
                    print("  -> Early stopping triggered.")
                    break

        best_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold+1}.pth")
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        test_correct = 0
        test_total = 0
        y_preds_test, y_trues_test = [], []

        with torch.inference_mode():
            for (x, y) in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                scores, predictions = model(x)

                preds_flat = predictions.reshape(-1)
                y_flat = y.reshape(-1)

                test_correct += (preds_flat == y_flat).sum().item()
                test_total += y_flat.size(0)

                y_preds_test.extend(preds_flat.cpu().numpy())
                y_trues_test.extend(y_flat.cpu().numpy())

        test_acc = 100.0 * test_correct / max(1, test_total)

        fold_accs['val'].append(best_val_acc)
        fold_accs['test'].append(test_acc)

        total_predictions.extend(y_preds_test)
        total_labels.extend(y_trues_test)

        print(f"Fold {fold+1} | Best Val Acc: {best_val_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        save_predictions_and_evaluate_fold1(
            y_trues_test, y_preds_test, class_names,
            out_dir, fold
        )

    val_scores = np.array(fold_accs['val'])
    test_scores = np.array(fold_accs['test'])

    print("\n================ FINAL RESULTS ================")
    print(f"Validation Accuracy: {val_scores.mean():.2f}% ± {val_scores.std():.2f}%")
    print(f"Test Accuracy:       {test_scores.mean():.2f}% ± {test_scores.std():.2f}%")

    save_predictions_and_evaluate(total_labels, total_predictions, class_names, out_dir)

if __name__ == "__main__":
    main()