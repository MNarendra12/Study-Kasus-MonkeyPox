import os
from collections import Counter
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Utils.getData import getImageLabel
from Models.model3.mpoxModel_mobilenetv3 import MpoxCNN

torch.set_num_threads(12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Threads: {torch.get_num_threads()} | Model: MobileNetV3-Small")

epochs = 30
batch_size = 32
lr = 3e-4

for fold in range(5, 6):
    print(f"\n{'='*45}\nTraining Fold {fold} (MobileNetV3-Small)\n{'='*45}")
    train_ds = getImageLabel(fold=fold, mode='train')
    val_ds = getImageLabel(fold=fold, mode='valid')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MpoxCNN(input_c=3, output=len(train_ds.classes), dropout=0.3).to(device)

    counts = Counter([label for _, label in train_ds.dataset])
    total = sum(counts.values())
    class_weights = torch.tensor(
        [total / (len(counts) * counts[i]) for i in range(len(counts))],
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    save_dir = f"Results/mobilenetv3/fold{fold}"
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

        train_loss = running_train_loss / len(train_ds)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                loss = criterion(model(images), labels)
                running_val_loss += loss.item() * images.size(0)

        val_loss = running_val_loss / len(val_ds)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']


        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1:02d}/{epochs}] - LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_dir}/best.pt")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, color='darkred', label='Training')
    plt.plot(range(1, epochs + 1), val_losses, color='dodgerblue', label='Validation')
    plt.title(f"Fold {fold} (MobileNetV3-Small) Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training.png")
    plt.close()
