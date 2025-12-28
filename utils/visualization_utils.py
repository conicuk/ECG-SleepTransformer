import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    class_accuracies = cm.diagonal() / cm.sum(axis=1) * 100
    print("Class-wise Accuracy:")
    for idx, acc in enumerate(class_accuracies):
        print(f"Class {idx} ({['W', 'Light_Sleep', 'Deep_Sleep', 'REM'][idx]}): {acc:.2f} %")
        #print(f"Class {idx} ({['W', 'N1', 'N2', 'N3','REM'][idx]}): {acc:.2f} %")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.4f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def print_classification_report(y_true, y_pred, class_names, save_path="classification_report.png"):
    report = classification_report(y_true, y_pred, digits=4, target_names=class_names)
    print("Classification Report:")
    print(report)

    plt.figure(figsize=(10, 6))
    plt.text(0.01, 0.99, str(report), {'fontsize': 12}, fontproperties='monospace', va='top')
    plt.axis('off')
    plt.subplots_adjust(top=0.8)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_class_performance(y_true, y_pred, class_names, save_path="class_performance.png"):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = {metric: [report[cls][metric] for cls in class_names] for metric in metrics}

    x = np.arange(len(class_names))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, class_metrics['precision'], width, label='Precision')
    plt.bar(x, class_metrics['recall'], width, label='Recall')
    plt.bar(x + width, class_metrics['f1-score'], width, label='F1-Score')

    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Class-wise Performance')
    plt.xticks(ticks=x, labels=class_names)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_hypnograms_selected_subjects(y_true, y_pred, subject_ids, class_names,
                                      selected_subjects, epoch_duration=30,
                                      save_dir="hypnograms", prefix="fold0"):
    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    subject_ids = np.array(subject_ids)

    for subj in selected_subjects:
        idx = subject_ids == subj
        if not np.any(idx):
            print(f"[Warning] Subject {subj} not found in subject_ids.")
            continue

        subj_y_true = y_true[idx]
        subj_y_pred = y_pred[idx]

        times = np.arange(len(subj_y_true) + 1) * epoch_duration / 3600  # 단위: 시간

        stages_true = np.append(subj_y_true, subj_y_true[-1])
        stages_pred = np.append(subj_y_pred, subj_y_pred[-1])

        fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

        axs[0].step(times, stages_true, where='post', color='brown', linewidth=1.0)
        axs[0].set_title("[Expert scoring]", fontsize=12)
        axs[0].invert_yaxis()

        axs[1].step(times, stages_pred, where='post', color='navy', linewidth=1.0)
        axs[1].set_title("[Model prediction]", fontsize=12)
        axs[1].invert_yaxis()

        for ax in axs:
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names)
            ax.set_ylim(-0.5, len(class_names) - 0.5)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.tick_params(labelsize=10)

        axs[1].set_xlabel("Time [hour]", fontsize=11)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{prefix}_subject_{subj}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"✅ Saved hypnogram for subject {subj} → {save_path}")

def plot_hypnogram_comparison(y_true, y_pred, class_names,
                              epoch_duration_sec=30,
                              save_path="hypnogram_comparison.png",
                              figsize=(14,6), dpi=150):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "y_true/y_pred 길이가 달라요!"

    times = np.arange(len(y_true)+1) * (epoch_duration_sec/60)

    stages_true = np.append(y_true, y_true[-1])
    stages_pred = np.append(y_pred, y_pred[-1])

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2,1, figsize=figsize, sharex=True)

    axes[0].step(times, stages_pred, where='post', color='navy', linewidth=1.0)
    axes[0].set_yticks(range(len(class_names)))
    axes[0].set_yticklabels(class_names)
    axes[0].invert_yaxis()
    axes[0].set_title("Model Prediction", fontsize=14)

    axes[1].step(times, stages_true, where='post', color='darkgreen', linewidth=1.0)
    axes[1].set_yticks(range(len(class_names)))
    axes[1].set_yticklabels(class_names)
    axes[1].invert_yaxis()
    axes[1].set_title("Expert Scoring", fontsize=14)

    axes[1].set_xlabel("Time (minutes)", fontsize=12)
    for ax in axes:
        ax.set_ylim(-0.5, len(class_names)-0.5)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"✅ Saved hypnogram comparison → {save_path}")


def plot_class_performance_with_std(y_true, y_pred, class_names, save_path="class_performance_with_std.png"):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']

    class_metrics = {metric: [report[cls][metric] for cls in class_names] for metric in metrics}

    class_mean_std = {
        metric: (np.mean(values), np.std(values)) for metric, values in class_metrics.items()
    }

    x = np.arange(len(class_names))
    width = 0.2

    # Plot the metrics
    plt.figure(figsize=(12, 8))
    plt.bar(x - width, class_metrics['precision'], width, label='Precision', yerr=class_mean_std['precision'][1])
    plt.bar(x, class_metrics['recall'], width, label='Recall', yerr=class_mean_std['recall'][1])
    plt.bar(x + width, class_metrics['f1-score'], width, label='F1-Score', yerr=class_mean_std['f1-score'][1])

    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Class-wise Performance (Mean ± Std)')
    plt.xticks(ticks=x, labels=class_names)
    plt.legend()

    for i, metric in enumerate(metrics):
        for j, value in enumerate(class_metrics[metric]):
            mean, std = class_mean_std[metric]
            plt.text(j + (i - 1) * width, value + 0.01, f"{value:.2f}\n±{std:.2f}", ha='center', fontsize=8)

    plt.savefig(save_path)
    plt.close()

    print("Detailed Performance:")
    for metric, (mean, std) in class_mean_std.items():
        print(f"{metric.capitalize()}: Mean = {mean:.2f}, Std = {std:.2f}")

def save_predictions_and_evaluate_fold1(y_true, y_pred, class_names, output_dir, fold):
    os.makedirs(output_dir, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=os.path.join(output_dir, f"confusion_matrix_fold_{fold}.png"))
    print_classification_report(y_true, y_pred, class_names, save_path=os.path.join(output_dir, f"classification_report_fold_{fold}.png"))
    plot_class_performance(y_true, y_pred, class_names, save_path=os.path.join(output_dir, f"class_performance_fold_{fold}.png"))


    kappa_score = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\nAdditional Statistics:")
    print(f"Cohen's Kappa fold {fold}: {kappa_score:.4f}")
    print(f"- Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    stats_file = os.path.join(output_dir, f"evaluation_statistics_fold{fold}.txt")
    with open(stats_file, 'w') as f:
        f.write("Additional Statistics:\n")
        f.write(f"- Cohen's Kappa: {kappa_score:.4f}\n")
        f.write(f"- Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
    print(f"Statistics saved to {stats_file}")

def save_total_labels_and_predictions(labels, predictions, out_dir):
    labels_path = os.path.join(out_dir, f"total_labels.npy")
    predictions_path = os.path.join(out_dir, f"total_predictions.npy")

    np.save(labels_path, labels)
    np.save(predictions_path, predictions)

    print(f"Total labels and predictions saved successfully:\n{labels_path}\n{predictions_path}")


def load_total_labels_and_predictions(out_dir):
    labels_path = os.path.join(out_dir, f"total_labels.npy")
    predictions_path = os.path.join(out_dir, f"total_predictions.npy")

    labels = np.load(labels_path)
    predictions = np.load(predictions_path)

    print(f"Total labels and predictions loaded successfully:\n{labels_path}\n{predictions_path}")
    return labels, predictions

def save_predictions_and_evaluate(y_true, y_pred, class_names, output_dir, epoch_duration=30):
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=os.path.join(output_dir, "confusion_matrix.png"))
    print_classification_report(y_true, y_pred, class_names, save_path=os.path.join(output_dir, "classification_report.png"))
    plot_class_performance(y_true, y_pred, class_names, save_path=os.path.join(output_dir, "class_performance.png"))
    save_total_labels_and_predictions(y_true, y_pred, output_dir)
    plot_class_performance_with_std(y_true, y_pred, class_names, save_path=os.path.join(output_dir, "lass_performance_with_std.png"))

    kappa_score = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\nAdditional Statistics:")
    print(f"- Cohen's Kappa: {kappa_score:.4f}")
    print(f"- Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    stats_file = os.path.join(output_dir, f"evaluation_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("Additional Statistics:\n")
        f.write(f"- Cohen's Kappa: {kappa_score:.4f}\n")
        f.write(f"- Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
    print(f"Statistics saved to {stats_file}")