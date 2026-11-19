import os
import json
from collections import Counter
import matplotlib.pyplot as plt
from Utils.getData import getImageLabel


def cek_distribusi_kelas(save_dir="Results"):
    ds_train = getImageLabel(fold=1, mode="train")
    ds_valid = getImageLabel(fold=1, mode="valid")
    ds_test = getImageLabel(fold=1, mode="test")
    class_names = ds_train.classes

    orig_train_counts = Counter(
        label for path, label in ds_train.dataset if "Original Images" in path
    )
    aug_train_counts = Counter(
        label for path, label in ds_train.dataset if "Augmented Images" in path
    )
    valid_counts = Counter(label for _, label in ds_valid.dataset)
    test_counts = Counter(label for _, label in ds_test.dataset)

    header = (
        f"{'Kelas':<20} | {'Orig Train':>10} | {'Aug Train':>10} | "
        f"{'Total Train':>11} | {'Valid':>7} | {'Test':>7} | {'Total All':>9}"
    )
    print(header)
    print("-" * len(header))

    totals = [0, 0, 0, 0, 0, 0]
    ringkasan = {}

    for i, name in enumerate(class_names):
        o_tr = orig_train_counts[i]
        a_tr = aug_train_counts[i]
        t_tr = o_tr + a_tr
        va = valid_counts[i]
        te = test_counts[i]
        t_all = t_tr + va + te

        ringkasan[name] = {
            "orig_train": o_tr,
            "aug_train": a_tr,
            "total_train": t_tr,
            "valid": va,
            "test": te,
            "total_all": t_all
        }

        vals = [o_tr, a_tr, t_tr, va, te, t_all]
        for idx, v in enumerate(vals):
            totals[idx] += v

        print(
            f"{name:<20} | {o_tr:>10} | {a_tr:>10} | "
            f"{t_tr:>11} | {va:>7} | {te:>7} | {t_all:>9}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<20} | {totals[0]:>10} | {totals[1]:>10} | "
        f"{totals[2]:>11} | {totals[3]:>7} | {totals[4]:>7} | {totals[5]:>9}"
    )

    eda_dir = os.path.join(save_dir, "EDA")
    os.makedirs(eda_dir, exist_ok=True)

    with open(os.path.join(eda_dir, "distribusi_kelas.json"), "w") as f:
        json.dump(ringkasan, f, indent=4)

    x = range(len(class_names))
    plt.figure(figsize=(10, 5))
    plt.bar([i - 0.25 for i in x], [orig_train_counts[i] for i in x], width=0.25, label="Original Train")
    plt.bar(x, [aug_train_counts[i] for i in x], width=0.25, label="Augmented Train")
    plt.bar([i + 0.25 for i in x], [valid_counts[i] + test_counts[i] for i in x], width=0.25, label="Valid + Test")
    plt.xticks(x, class_names, rotation=25, ha="right")
    plt.ylabel("Jumlah Gambar")
    plt.title("Distribusi Dataset per Kelas (Fold 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "distribusi_kelas.png"), dpi=300)
    plt.close()
    print(f"\nHasil tersimpan di folder: {eda_dir}/")


if __name__ == "__main__":
    cek_distribusi_kelas(save_dir="Results")