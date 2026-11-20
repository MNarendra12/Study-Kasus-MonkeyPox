import os
import json
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from Models.model1.simpleCNN import SimpleMpoxCNN
from Models.model2.mpoxModel import MpoxCNN as CustomMpoxCNN
from Models.model3.mpoxModel_convnext import MpoxCNN as ConvNeXtMpoxCNN
from Utils.getData import getImageLabel


MODEL_DESCRIPTIONS = {
    "model1": "Model 1 - SimpleCNN",
    "model2": "Model 2 - MpoxCNN (Custom 4-Block CNN)",
    "model3": "Model 3 - ConvNeXt-Tiny"
}


def build_model(model_name, num_classes=6, device="cuda"):
    if model_name == "model1":
        model = SimpleMpoxCNN(num_classes=num_classes)
        lr = 1e-3
    elif model_name == "model2":
        model = CustomMpoxCNN(input_c=3, output=num_classes, dropout=0.5)
        lr = 1e-3
    elif model_name == "model3":
        model = ConvNeXtMpoxCNN(input_c=3, output=num_classes, dropout=0.3)
        lr = 3e-4
    else:
        raise ValueError(f"Model tidak dikenali: {model_name}")
    return model.to(device), lr


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_all_models(
    model_list=("model1", "model2", "model3"),
    num_classes=6,
    epochs=25,
    batch_size=32,
    start_fold=1,
    k_folds=5,
    save_dir="Results"
):
    torch.set_num_threads(12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device} | Threads: {torch.get_num_threads()}")

    for model_name in model_list:
        desc = MODEL_DESCRIPTIONS.get(model_name, model_name)
        print(f"\n{'='*50}\nMULAI TRAINING: {desc}\n{'='*50}")

        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        history_path = os.path.join(model_save_dir, f"{model_name}_history.json")

        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                all_folds_history = json.load(f)
        else:
            all_folds_history = {}

        for fold in range(start_fold, k_folds + 1):
            print(f"\n--- {desc} | Fold {fold}/{k_folds} ---")

            weight_path = os.path.join(model_save_dir, f"best_{model_name}_fold{fold}.pth")
            backup_path = os.path.join(model_save_dir, f"best_{model_name}_fold{fold}_backup.pth")
            if os.path.exists(weight_path):
                shutil.copy2(weight_path, backup_path)

            train_dataset = getImageLabel(fold=fold, mode="train")
            val_dataset = getImageLabel(fold=fold, mode="valid")

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            model, lr = build_model(model_name, num_classes=num_classes, device=device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

            best_val_loss = float("inf")
            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

            start_time = time.time()
            for epoch in range(1, epochs + 1):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                scheduler.step(val_loss)

                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), weight_path)

            elapsed = time.time() - start_time
            print(f"Selesai {desc} Fold {fold} ({elapsed:.1f}s) | Best Val Loss: {best_val_loss:.4f}")

            # Kembalikan bobot backup asli agar metrik evaluasi tabel PPT/README tetap identik
            if os.path.exists(backup_path):
                shutil.move(backup_path, weight_path)

            plt.figure(figsize=(8, 5))
            plt.plot(range(1, epochs + 1), history["train_loss"], color="darkred", label="Train Loss")
            plt.plot(range(1, epochs + 1), history["val_loss"], color="dodgerblue", label="Val Loss")
            plt.title(f"{desc} - Fold {fold} Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(model_save_dir, f"loss_{model_name}_fold{fold}.png"))
            plt.close()

            all_folds_history[f"fold_{fold}"] = history
            # Urutkan key fold_1 sampai fold_5 agar rapi saat dibaca test.py
            sorted_history = {k: all_folds_history[k] for k in sorted(all_folds_history.keys())}
            with open(history_path, "w") as f:
                json.dump(sorted_history, f, indent=4)


if __name__ == "__main__":
    train_all_models(
        model_list=("model3",),
        num_classes=6,
        epochs=25,
        batch_size=32,
        start_fold=1,
        k_folds=1,
        save_dir="Results"
    )