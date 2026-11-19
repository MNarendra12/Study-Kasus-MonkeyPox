import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from Models.model1.simpleCNN import SimpleMpoxCNN
from Models.model2.mpoxModel import MpoxCNN as CustomMpoxCNN
from Models.model3.mpoxModel_convnext import MpoxCNN as ConvNeXtMpoxCNN
from Utils.getData import getImageLabel


MODEL_DESCRIPTIONS = {
    "model1": "Model 1 - SimpleCNN",
    "model2": "Model 2 - MpoxCNN (Custom 4-Block CNN)",
    "model3": "Model 3 - ConvNeXt-Tiny"
}


def build_model(model_name, num_classes=6, device="cpu"):
    if model_name == "model1":
        model = SimpleMpoxCNN(num_classes=num_classes)
    elif model_name == "model2":
        model = CustomMpoxCNN(input_c=3, output=num_classes, dropout=0.5)
    elif model_name == "model3":
        model = ConvNeXtMpoxCNN(input_c=3, output=num_classes, dropout=0.3)
    else:
        raise ValueError(f"Model tidak dikenali: {model_name}")
    return model.to(device)


def plot_training_curves(model_name, model_dir):
    history_path = os.path.join(model_dir, f"{model_name}_history.json")
    if not os.path.exists(history_path):
        print(f"History {history_path} tidak ditemukan, grafik kurva dilewati.")
        return

    with open(history_path, "r") as f:
        all_folds = json.load(f)

    desc = MODEL_DESCRIPTIONS.get(model_name, model_name)
    folds = sorted(all_folds.keys())
    num_folds = len(folds)

    fig, axes = plt.subplots(2, num_folds, figsize=(4 * num_folds, 7), sharex=True)
    all_train_loss, all_val_loss = [], []
    all_train_acc, all_val_acc = [], []

    for idx, fold_key in enumerate(folds):
        h = all_folds[fold_key]
        epochs = range(1, len(h["train_loss"]) + 1)

        all_train_loss.append(h["train_loss"])
        all_val_loss.append(h["val_loss"])
        all_train_acc.append(h["train_acc"])
        all_val_acc.append(h["val_acc"])

        ax_loss = axes[0, idx] if num_folds > 1 else axes[0]
        ax_loss.plot(epochs, h["train_loss"], label="Train Loss", color="#1f77b4", linewidth=1.8)
        ax_loss.plot(epochs, h["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=1.8)
        ax_loss.set_title(f"Fold {idx + 1}")
        ax_loss.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax_loss.set_ylabel("Loss")
        if idx == num_folds - 1:
            ax_loss.legend()

        ax_acc = axes[1, idx] if num_folds > 1 else axes[1]
        ax_acc.plot(epochs, h["train_acc"], label="Train Acc", color="#2ca02c", linewidth=1.8)
        ax_acc.plot(epochs, h["val_acc"], label="Val Acc", color="#d62728", linewidth=1.8)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax_acc.set_ylabel("Accuracy")
        if idx == num_folds - 1:
            ax_acc.legend()

    fig.suptitle(f"Kurva Training & Validasi per Fold - {desc}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, f"{model_name}_kurva_5fold.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Grafik Rata-Rata 5 Fold (Mean +/- Std)
    mean_train_loss, std_train_loss = np.mean(all_train_loss, axis=0), np.std(all_train_loss, axis=0)
    mean_val_loss, std_val_loss = np.mean(all_val_loss, axis=0), np.std(all_val_loss, axis=0)
    mean_train_acc, std_train_acc = np.mean(all_train_acc, axis=0), np.std(all_train_acc, axis=0)
    mean_val_acc, std_val_acc = np.mean(all_val_acc, axis=0), np.std(all_val_acc, axis=0)

    epochs = range(1, len(mean_train_loss) + 1)
    fig_mean, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, mean_train_loss, label="Mean Train Loss", color="#1f77b4", linewidth=2)
    ax1.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color="#1f77b4", alpha=0.15)
    ax1.plot(epochs, mean_val_loss, label="Mean Val Loss", color="#ff7f0e", linewidth=2)
    ax1.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color="#ff7f0e", alpha=0.15)
    ax1.set_title(f"Rata-rata Loss (5-Fold) - {desc}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs, mean_train_acc, label="Mean Train Acc", color="#2ca02c", linewidth=2)
    ax2.fill_between(epochs, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, color="#2ca02c", alpha=0.15)
    ax2.plot(epochs, mean_val_acc, label="Mean Val Acc", color="#d62728", linewidth=2)
    ax2.fill_between(epochs, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, color="#d62728", alpha=0.15)
    ax2.set_title(f"Rata-rata Accuracy (5-Fold) - {desc}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig_mean.tight_layout()
    fig_mean.savefig(os.path.join(model_dir, f"{model_name}_kurva_ratarata.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_mean)


def evaluate_model(model_name, save_dir="Results", batch_size=64, k_folds=5, device="cpu"):
    desc = MODEL_DESCRIPTIONS.get(model_name, model_name)
    model_dir = os.path.join(save_dir, model_name)

    if not os.path.exists(model_dir):
        print(f"Folder {model_dir} belum ditemukan. Dilewati.")
        return None

    print(f"EVALUASI TESTING: {desc}")

    plot_training_curves(model_name, model_dir)

    fold_metrics = []
    all_y_true, all_y_pred = [], []
    class_names = None

    for fold in range(1, k_folds + 1):
        weight_path = os.path.join(model_dir, f"best_{model_name}_fold{fold}.pth")
        if not os.path.exists(weight_path):
            print(f"Bobot {weight_path} tidak ditemukan, fold {fold} dilewati.")
            continue

        test_dataset = getImageLabel(fold=fold, mode="test")
        if class_names is None:
            class_names = test_dataset.classes

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model = build_model(model_name, num_classes=len(class_names), device=device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()

        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

        fold_metrics.append({
            "Model": desc,
            "Fold": f"Fold {fold}",
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        # Simpan Confusion Matrix per fold
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{desc} - Confusion Matrix (Fold {fold})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f"cm_{model_name}_fold{fold}.png"), dpi=300)
        plt.close()

    if not fold_metrics:
        return None

    df_folds = pd.DataFrame(fold_metrics)
    print("\n--- Hasil Evaluasi Per Fold ---")
    print(df_folds.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Hitung Rata-Rata & Standar Deviasi 5 Fold
    summary_stats = {
        "Model": desc,
        "Accuracy": f"{df_folds['Accuracy'].mean():.4f} +/- {df_folds['Accuracy'].std():.4f}",
        "Precision": f"{df_folds['Precision'].mean():.4f} +/- {df_folds['Precision'].std():.4f}",
        "Recall": f"{df_folds['Recall'].mean():.4f} +/- {df_folds['Recall'].std():.4f}",
        "F1-Score": f"{df_folds['F1-Score'].mean():.4f} +/- {df_folds['F1-Score'].std():.4f}",
        "ROC-AUC": f"{df_folds['ROC-AUC'].mean():.4f} +/- {df_folds['ROC-AUC'].std():.4f}"
    }

    print("\n--- Rata-Rata 5-Fold Cross Validation ---")
    for k, v in summary_stats.items():
        if k != "Model":
            print(f"{k:<10}: {v}")

    # Simpan tabel per-fold ke CSV
    df_folds.to_csv(os.path.join(model_dir, f"{model_name}_hasil_5fold.csv"), index=False)

    # Classification Report Gabungan 5 Fold (Evaluasi Per Kelas untuk PPT)
    print("\n--- Performa Per Kelas (Gabungan 5 Fold Test Set) ---")
    report_text = classification_report(all_y_true, all_y_pred, target_names=class_names, digits=4, zero_division=0)
    print(report_text)

    report_dict = classification_report(all_y_true, all_y_pred, target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(model_dir, f"{model_name}_per_kelas.csv"))

    # Confusion Matrix Gabungan Seluruh 5 Fold (Sangat jelas untuk analisis kesalahan kelas di PPT)
    cm_total = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{desc} - Total Confusion Matrix (5 Fold)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"cm_{model_name}_total_5fold.png"), dpi=300)
    plt.close()

    return summary_stats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device evaluasi: {device}\n")

    model_list = ("model1", "model2", "model3")
    all_summaries = []

    for model_name in model_list:
        summary = evaluate_model(model_name, save_dir="Results", batch_size=64, k_folds=5, device=device)
        if summary is not None:
            all_summaries.append(summary)

    if all_summaries:
        df_compare = pd.DataFrame(all_summaries)
        print("\n=== PERBANDINGAN RATA-RATA 5-FOLD ANTAR MODEL ===")
        print(df_compare.to_string(index=False))
        df_compare.to_csv("Results/perbandingan_semua_model.csv", index=False)


if __name__ == "__main__":
    main()