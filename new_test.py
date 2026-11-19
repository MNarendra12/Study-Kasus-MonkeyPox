import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from Models.model1.simpleCNN import SimpleMpoxCNN
from Models.model2.mpoxModel import MpoxCNN as CustomMpoxCNN
from Models.model3.mpoxModel_mobilenetv3 import MpoxCNN as MobileNetV3MpoxCNN
from Utils.getData import getImageLabel


MODEL_DESCRIPTIONS = {
    "model1": "Model 1 - SimpleCNN",
    "model2": "Model 2 - MpoxCNN (Custom 4-Block CNN)",
    "model3": "Model 3 - MobileNetV3-Small"
}


def load_trained_model(model_name, weight_path, num_classes=6, device="cpu"):
    if model_name == "model1":
        model = SimpleMpoxCNN(num_classes=num_classes)
    elif model_name == "model2":
        model = CustomMpoxCNN(input_c=3, output=num_classes, dropout=0.5)
    elif model_name == "model3":
        model = MobileNetV3MpoxCNN(input_c=3, output=num_classes, dropout=0.3)
    else:
        raise ValueError(f"Model tidak dikenali: {model_name}")

    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_dataset(model, dataloader, device):
    all_preds = []
    all_probs = []
    all_labels = []
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def save_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_all_models(model_list=("model1", "model2", "model3"), k_folds=5, batch_size=32, save_dir="Results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_results = {}

    for model_name in model_list:
        desc = MODEL_DESCRIPTIONS.get(model_name, model_name)
        print("=" * 75)
        print(f"EVALUASI TEST SET: {desc}")
        print("=" * 75)

        model_dir = os.path.join(save_dir, model_name)
        fold_metrics = []
        total_cm = None
        all_folds_true = []
        all_folds_pred = []
        class_names = None

        for fold in range(1, k_folds + 1):
            weight_path = os.path.join(model_dir, f"best_{model_name}_fold{fold}.pth")
            if not os.path.exists(weight_path):
                print(f"[SKIP] Bobot tidak ditemukan: {weight_path}")
                continue

            test_dataset = getImageLabel(fold=fold, mode="test")
            if class_names is None:
                class_names = getattr(test_dataset, "classes", [f"Class {i}" for i in range(6)])

            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            model = load_trained_model(model_name, weight_path, num_classes=len(class_names), device=device)

            y_true, y_pred, y_prob = predict_dataset(model, test_loader, device)
            all_folds_true.extend(y_true)
            all_folds_pred.extend(y_pred)

            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
            total_cm = cm if total_cm is None else total_cm + cm

            fold_metrics.append({
                "fold": fold,
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "roc_auc": float(auc)
            })

            print(
                f"Fold {fold} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
                f"Recall: {rec:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}"
            )

        if not fold_metrics:
            continue

        accs = [m["accuracy"] for m in fold_metrics]
        precs = [m["precision"] for m in fold_metrics]
        recs = [m["recall"] for m in fold_metrics]
        f1s = [m["f1_score"] for m in fold_metrics]
        aucs = [m["roc_auc"] for m in fold_metrics]

        print("-" * 75)
        print(f"RATA-RATA 5 FOLD ({desc}):")
        print(f"Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
        print(f"Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
        print(f"F1-Score  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"ROC-AUC   : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
        print("-" * 75)

        print(f"LAPORAN PER KELAS (AKUMULASI 5 FOLD - {desc}):")
        print(classification_report(all_folds_true, all_folds_pred, target_names=class_names, digits=4, zero_division=0))

        cm_path = os.path.join(model_dir, f"{model_name}_confusion_matrix.png")
        save_confusion_matrix(
            total_cm,
            class_names,
            title=f"Confusion Matrix (5-Fold Total) - {desc}",
            save_path=cm_path
        )

        summary_results[model_name] = {
            "per_fold": fold_metrics,
            "mean_std": {
                "accuracy": f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
                "precision": f"{np.mean(precs):.4f} ± {np.std(precs):.4f}",
                "recall": f"{np.mean(recs):.4f} ± {np.std(recs):.4f}",
                "f1_score": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
                "roc_auc": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
            }
        }

    with open(os.path.join(save_dir, "test_evaluation_summary.json"), "w") as f:
        json.dump(summary_results, f, indent=4)
    print("\nEvaluasi selesai. Semua ringkasan & Confusion Matrix tersimpan di folder Results/.")


if __name__ == "__main__":
    evaluate_all_models(model_list=("model1", "model2", "model3"), k_folds=5, batch_size=32, save_dir="Results")
