from torch.utils.data import DataLoader
from Utils.getData import getImageLabel

train_ds = getImageLabel(fold=1, mode='train')
val_ds = getImageLabel(fold=1, mode='valid')
test_ds = getImageLabel(fold=1, mode='test')

print("Classes:", train_ds.classes)
print(f"Jumlah data -> Train: {len(train_ds)}, Valid: {len(val_ds)}, Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
images, labels = next(iter(train_loader))

print("Batch images shape:", images.shape)
print("Batch labels shape:", labels.shape)
print("Rentang nilai piksel (setelah normalisasi):", round(images.min().item(), 2), "s.d.", round(images.max().item(), 2))
print("Rentang label:", labels.min().item(), "s.d.", labels.max().item())