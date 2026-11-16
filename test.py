import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from Utils.getData import getImageLabel
from Models.mpoxModel import MpoxCNN


def main():
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['Chickenpox', 'Cowpox', 'Healthy', 'HFMD', 'Measles', 'Mpox']

    acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []

    for fold in range(1, 6):
        print(f"TESTING FOLD {fold}")
        os.makedirs(f"./Results/fold{fold}", exist_ok=True)

        test_loader = DataLoader(getImageLabel(fold=fold, mode='test'), batch_size=BATCH_SIZE, shuffle=False)

        model = MpoxCNN(input_c=3, output=6, dropout=0.5).to(device)
        model.load_state_dict(torch.load(f"./Results/fold{fold}/best.pt", map_location=device, weights_only=True))
        model.eval()

        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for src, trg in test_loader:
                src = src.to(device)
                logits = model(src)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                y_true.extend(trg.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        auc_list.append(auc)

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"ROC-AUC   : {auc:.4f}\n")

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Fold {fold} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(f"./Results/fold{fold}/confusion_matrix.png")
        plt.close()

    print("RATA-RATA 5-FOLD CROSS VALIDATION")
    print(f"Mean Accuracy  : {np.mean(acc_list):.4f} +/- {np.std(acc_list):.4f}")
    print(f"Mean Precision : {np.mean(prec_list):.4f} +/- {np.std(prec_list):.4f}")
    print(f"Mean Recall    : {np.mean(rec_list):.4f} +/- {np.std(rec_list):.4f}")
    print(f"Mean F1-Score  : {np.mean(f1_list):.4f} +/- {np.std(f1_list):.4f}")
    print(f"Mean ROC-AUC   : {np.mean(auc_list):.4f} +/- {np.std(auc_list):.4f}")


if __name__ == "__main__":
    main()