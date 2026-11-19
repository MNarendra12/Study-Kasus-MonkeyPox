import os
import numpy as np
import matplotlib.pyplot as plt

models = [
    "Model 1\n(SimpleCNN)",
    "Model 2\n(Custom 4-Block CNN)",
    "Model 3\n(ConvNeXt-Tiny)"
]

data = {
    "Accuracy": [0.5381, 0.6332, 0.8120],
    "Precision": [0.4692, 0.5745, 0.8188],
    "Recall": [0.3897, 0.5325, 0.7640],
    "F1-Score": [0.3717, 0.5354, 0.7736],
    "ROC-AUC": [0.7677, 0.8517, 0.9722]
}

x = np.arange(len(models))
width = 0.15

fig, ax = plt.subplots(figsize=(11, 5.5))

for idx, (metric, values) in enumerate(data.items()):
    offset = (idx - 2) * width
    bars = ax.bar(x + offset, values, width, label=metric)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.012,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Skor Rata-Rata (5-Fold)")
ax.set_title("Perbandingan Performa Rata-Rata 5-Fold Antar Model")
ax.legend(loc="upper left", ncol=5)
ax.grid(axis="y", linestyle="--", alpha=0.4)

os.makedirs("Results", exist_ok=True)
plt.tight_layout()
plt.savefig("Results/perbandingan_model.png", dpi=300)
plt.close()
print("Grafik tersimpan di Results/perbandingan_model.png")