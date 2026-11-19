import os
import matplotlib.pyplot as plt

# Data asli dari log MobileNetV3-Small Fold 5/5 (Epoch 1 - 25)
epochs = list(range(1, 26))

train_loss = [
    0.9960, 0.7252, 0.6296, 0.5653, 0.5109, 0.4782, 0.4313, 0.3994, 0.3737, 0.3568,
    0.3305, 0.3297, 0.3079, 0.2954, 0.2975, 0.2947, 0.2860, 0.2847, 0.2787, 0.2683,
    0.2775, 0.2693, 0.2747, 0.2754, 0.2661
]

val_loss = [
    1.0489, 0.9893, 1.0375, 0.9886, 0.9798, 0.9839, 1.0337, 1.0298, 1.0488, 1.0194,
    1.0631, 1.0825, 1.0724, 1.0816, 1.0665, 1.0928, 1.0896, 1.0687, 1.0741, 1.0541,
    1.0642, 1.0607, 1.0690, 1.0868, 1.0905
]

train_acc = [
    0.6343, 0.7344, 0.7651, 0.7955, 0.8149, 0.8304, 0.8447, 0.8549, 0.8690, 0.8715,
    0.8851, 0.8822, 0.8934, 0.8996, 0.8937, 0.8963, 0.9001, 0.9019, 0.9025, 0.9041,
    0.9083, 0.9118, 0.9058, 0.9042, 0.9071
]

val_acc = [
    0.6818, 0.6429, 0.6299, 0.6688, 0.6623, 0.7078, 0.6818, 0.6818, 0.7013, 0.6818,
    0.6818, 0.6883, 0.6883, 0.6688, 0.6688, 0.6688, 0.6688, 0.6494, 0.6688, 0.6688,
    0.6818, 0.6623, 0.6753, 0.6753, 0.6558
]

plt.figure(figsize=(12, 5))

# Subplot 1: Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('MobileNetV3-Small Fold 5 - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Subplot 2: Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Val Accuracy')
plt.title('MobileNetV3-Small Fold 5 - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()

# Ganti path di bawah ini jika folder fold5 kamu berada di dalam subfolder lain (misal: Results/MobileNetV3-Small/fold5/training.png)
output_path = 'Results/fold5/training.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"Grafik berhasil disimpan di: {output_path}")
