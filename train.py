import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from Utils.getData import getImageLabel
from Models.mpoxModel import MpoxCNN


def main():
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(12)
    print(f"Device: {device} | Threads: {torch.get_num_threads()}")

    for fold in range(1, 6):
        print(f"TRAINING FOLD {fold}")
        os.makedirs(f"./Results/fold{fold}", exist_ok=True)

        train_loader = DataLoader(getImageLabel(fold=fold, mode='train'), batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(getImageLabel(fold=fold, mode='valid'), batch_size=BATCH_SIZE, shuffle=False)

        model = MpoxCNN(input_c=3, output=6, dropout=0.5).to(device)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.CrossEntropyLoss()

        loss_train_all, loss_valid_all = [], []
        best_val_loss = float('inf')

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for src, trg in train_loader:
                src, trg = src.to(device), trg.to(device)
                optimizer.zero_grad()
                pred = model(src)
                loss = loss_function(pred, trg)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for src, trg in valid_loader:
                    src, trg = src.to(device), trg.to(device)
                    pred = model(src)
                    loss = loss_function(pred, trg)
                    valid_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_valid_loss = valid_loss / len(valid_loader)
            loss_train_all.append(avg_train_loss)
            loss_valid_all.append(avg_valid_loss)

            print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")

            if avg_valid_loss < best_val_loss:
                best_val_loss = avg_valid_loss
                torch.save(model.state_dict(), f"./Results/fold{fold}/best.pt")

        plt.figure()
        plt.plot(range(1, EPOCHS + 1), loss_train_all, color="#931a00", label='Training')
        plt.plot(range(1, EPOCHS + 1), loss_valid_all, color="#3399e6", label='Validation')
        plt.title(f"Fold {fold} Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"./Results/fold{fold}/training.png")
        plt.close()


if __name__ == "__main__":
    main()