import torch.optim as optim
import time
import yaml
import os
import sys
import random
import gc

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from torch.utils.data import TensorDataset
from data_loader.data_loaders import *
from utils.visualization_utils import save_predictions_and_evaluate, save_predictions_and_evaluate_fold1

from models.Baseline import *
from sklearn.model_selection import KFold, train_test_split
from models.Proposed_Model import *
from utils.util import *
from tqdm import tqdm

def main():
    gc.collect()
    torch.cuda.empty_cache()
    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/sleep_stage.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # 랜덤 시드 세팅
    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    torch.backends.cudnn.benchmark = True

    if params['task'] == "shhs1" or params['task'] == "shhs2":
        total_data = load_data_shhs(params['data_files'], params['task'])

    elif params['task'] == "isruc1":
        total_data = load_data_isruc_s1(params['data_files'])

    elif params['task'] == "mesa":
        total_data = load_data_mesa(params['data_files'])

    kfolds = params['k_folds']

    if params['data'] == "ECG_IHR" or params['data'] == "Mel":
        if params['data'] == "ECG_IHR":
            data_x, data_y = data_generator_np(total_data, 'ECG_IHR')
        elif params['data'] == "Mel":
            data_x, data_y = data_generator_np(total_data, 'Mel')
        print('The number of x data: ', len(data_x), data_x.shape)
        print(data_x[0])
        print(data_x.shape)

        dataset = TensorDataset(data_x, data_y)

    if params['data'] == "IHR_EDR":
        data_ihr, data_edr, data_y = data_generator_np_IHR_EDR(total_data)
        print('The number of IHR data: ', len(data_ihr), data_ihr.shape)
        print(data_ihr.shape)

        print('The number of EDR data: ', len(data_edr), data_edr.shape)
        print(data_edr.shape)

        print('The number of labels: ', len(data_y), data_y.shape)

        dataset = TensorDataset(data_ihr, data_edr, data_y)

    kfold = KFold(n_splits=kfolds, shuffle=True, random_state=params['random_seed'])

    criterion = nn.CrossEntropyLoss()

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    fold_accs = {'val': [], 'test': []}
    total_predictions = []
    total_labels = []

    # training
    start_time = time.time()

    print('========================================')
    print("Start training...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):

        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        train_subset_idx, val_subset_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_subset_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_subset_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], sampler=train_subsampler, num_workers=4)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], sampler=val_subsampler, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], sampler=test_subsampler, num_workers=4)

        print(f"Fold {fold}:")
        print(f"Trainset size: {len(train_subset_idx)}")
        print(f"Validationset size: {len(val_subset_idx)}")
        print(f"Test size: {len(val_idx)}")

        model = Proposed()  #Baseline1() <-Raw ECG, Baseline2() <- IHR_EDR, Baseline3() <- IHR, Proposed() <- Mel-spectrogram #<- change model
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=params['optimizer_params'][params['optimizer']]['lr'])

        best_val_acc = 0

        for epoch in range(params['max_epochs']):
            global_steps = 0
            train_loss = 0
            train_correct_cnt = 0
            train_batch_cnt = 0
            model.train()
            data_iter = tqdm(train_loader, desc=f"Fold {fold + 1}, EP:{epoch + 1}")

            for x, y in data_iter:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                outputs = model(x)

            # for x, x2, y in data_iter: # If you want use Baseline2
            #     x = x.to(device)
            #     x2 = x2.to(device)
            #     y = y.to(device)
            #     optimizer.zero_grad()
            #     outputs = model(x, x2)


                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()


                train_loss += loss.item()
                train_batch_cnt += 1

                _, top_pred = torch.topk(outputs, k=1, dim=-1)
                top_pred = top_pred.squeeze(dim=1)
                train_correct_cnt += int(torch.sum(top_pred == y))

                global_steps += 1

            train_acc = train_correct_cnt * 100 / len(train_subset_idx)
            train_ave_loss = train_loss / len(data_iter)
            training_time = (time.time() - start_time) / 60

            print('========================================')
            print("epoch:", epoch, "/ global_steps:", global_steps)
            print("training dataset average loss: %.4f" % train_ave_loss)
            print(f"Training accuracy: {train_acc:.2f}%")
            print("training_time: %.2f minutes" % training_time)

            # validation (for early stopping)
            val_correct_cnt = 0
            val_loss = 0
            val_batch_cnt = 0

            model.eval()

            with torch.inference_mode():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    outputs = model(x)


                # for x, x2, y in val_loader: # If you want use Baseline2
                #     x = x.to(device)
                #     x2 = x2.to(device)
                #     y = y.to(device)
                #     outputs = model(x, x2)



                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    val_batch_cnt += y.size(0)

                    _, top_pred = torch.topk(outputs, k=1, dim=-1)
                    top_pred = top_pred.squeeze(dim=1)

                    val_correct_cnt += int(torch.sum(top_pred == y))

            val_acc = 100 * val_correct_cnt / val_batch_cnt
            val_ave_loss = val_loss / val_batch_cnt

            train_loss_list.append(train_ave_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_ave_loss)
            val_acc_list.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')
                counter = 0
            else:
                counter += 1
                print(f"EarlyStopping counter: {counter}/{params['patience']}")
                if counter >= params['patience']:
                    print("Early stopping triggered!")
                    break
            print(f"Validation accuracy: {val_acc:.4f}%")

        print('Training process has finished. Saving trained model.')

        print('Starting testing')

        # model.load_state_dict(torch.load(f'best_model_fold_{fold + 1}.pth'))
        # model.eval()
        # correct_cnt = 0
        # test_total = 0
        #
        # y_preds_test = []
        # y_trues_test = []

        best_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold+1}.pth")
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        test_correct = 0
        test_total = 0
        y_preds_test = []
        y_trues_test = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)

            # for x, x2, y in test_loader: # If you want use Baseline2
            #     x = x.to(device)
            #     x2 = x2.to(device)
            #     y = y.to(device)
            #     pred = model(x, x2)

                _, top_pred = torch.topk(pred, k=1, dim=-1)
                top_pred = top_pred.squeeze(dim=1)

                test_correct += int(torch.sum(top_pred == y))
                test_total += y.size(0)

                y_preds_test.extend(top_pred.cpu().numpy())
                y_trues_test.extend(y.cpu().numpy())

                total_predictions.extend(top_pred.cpu().numpy())
                total_labels.extend(y.cpu().numpy())

        test_acc = 100 * test_correct / test_total
        fold_accs['val'].append(best_val_acc)
        fold_accs['test'].append(test_acc)

        print(f"\nFold {fold + 1} Results:")
        print(f"Validation Accuracy: {best_val_acc:.2f}%")
        print('Test Accuracy for fold %d: %.2f %%' % (fold, test_acc))
        print('--------------------------------')

        save_predictions_and_evaluate_fold1(y_trues_test, y_preds_test, ["W", "Light_Sleep", "Deep_Sleep", "REM"], # 4-class
                                            out_dir, fold)
        # save_predictions_and_evaluate_fold1(y_trues_test, y_preds_test, ["W", "N1", "N2","N3", "REM"], # 5-class
        #                                     out_dir, fold)

    val_scores = np.array(fold_accs['val'])
    test_scores = np.array(fold_accs['test'])

    print("\nFinal Results:")
    print(f"Validation Accuracy: {val_scores.mean():.2f}% ± {val_scores.std():.2f}%")
    print(f"Test Accuracy: {test_scores.mean():.2f}% ± {test_scores.std():.2f}%")
    print('Each Fold test accuracy', test_scores)

    save_predictions_and_evaluate(total_labels, total_predictions, ["W", "Light_Sleep", "Deep_Sleep", "REM"], out_dir) # 4-class
    # save_predictions_and_evaluate(total_labels, total_predictions, ["W", "N1", "N2","N3", "REM"], out_dir) # 5-class

if __name__ == '__main__':
    main()