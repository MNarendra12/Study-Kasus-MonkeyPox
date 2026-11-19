import os
import json
from collections import Counter
import matplotlib.pyplot as plt
from Utils.getData import getImageLabel


def cek_distribusi_kelas(save_dir="Results", k_folds=5):
    eda_dir = os.path.join(save_dir, "EDA")
    os.makedirs(eda_dir, exist_ok=True)

    ringkasan_semua_fold = {}
    fig_all, axes_all = plt.subplots(1, k_folds, figsize=(4.5 * k_folds, 4.5), sharey=True)

    for idx_f, fold in enumerate(range(1, k_folds + 1)):
        ds_train = getImageLabel(fold=fold, mode="train")
        ds_valid = getImageLabel(fold=fold, mode="valid")
        ds_test = getImageLabel(fold=fold, mode="test")
        class_names = ds_train.classes

        orig_train_counts = Counter(
            label for path, label in ds_train.dataset if "Original Images" in path
        )
        aug_train_counts = Counter(
            label for path, label in ds_train.dataset if "Augmented Images" in path
        )
        valid_counts = Counter(label for _, label in ds_valid.dataset)
        test_counts = Counter(label for _, label in ds_test.dataset)

        print(f"\n=== DISTRIBUSI DATASET FOLD {fold} ===")
        header = (
            f"{'Kelas':<20} | {'Orig Train':>10} | {'Aug Train':>10} | "
            f"{'Total Train':>11} | {'Valid':>7} | {'Test':>7} | {'Total All':>9}"
        )
        print(header)
        print("-" * len(header))

        totals = [0, 0, 0, 0, 0, 0]
        ringkasan_fold = {}

        for i, name in enumerate(class_names):
            o_tr = orig_train_counts[i]
            a_tr = aug_train_counts[i]
            t_tr = o_tr + a_tr
            va = valid_counts[i]
            te = test_counts[i]
            t_all = t_tr + va + te

            ringkasan_fold[name] = {
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

        ringkasan_semua_fold[f"fold_{fold}"] = ringkasan_fold

        # 1. Simpan grafik individual per fold
        x = range(len(class_names))
        plt.figure(figsize=(10, 5))
        plt.bar([i - 0.25 for i in x], [orig_train_counts[i] for i in x], width=0.25, label="Original Train")
        plt.bar(x, [aug_train_counts[i] for i in x], width=0.25, label="Augmented Train")
        plt.bar([i + 0.25 for i in x], [valid_counts[i] + test_counts[i] for i in x], width=0.25, label="Valid + Test")
        plt.xticks(x, class_names, rotation=25, ha="right")
        plt.ylabel("Jumlah Gambar")
        plt.title(f"Distribusi Dataset per Kelas (Fold {fold})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, f"distribusi_kelas_fold{fold}.png"), dpi=300)
        plt.close()

        # 2. Isi subplot untuk gambar gabungan semua fold
        ax = axes_all[idx_f]
        ax.bar([i - 0.25 for i in x], [orig_train_counts[i] for i in x], width=0.25, label="Original Train")
        ax.bar(x, [aug_train_counts[i] for i in x], width=0.25, label="Augmented Train")
        ax.bar([i + 0.25 for i in x], [valid_counts[i] + test_counts[i] for i in x], width=0.25, label="Valid + Test")
        ax.set_xticks(list(x))
        ax.set_xticklabels(class_names, rotation=35, ha="right")
        ax.set_title(f"Fold {fold}")
        if idx_f == 0:
            ax.set_ylabel("Jumlah Gambar")
        if idx_f == k_folds - 1:
            ax.legend(loc="upper right")

    with open(os.path.join(eda_dir, "distribusi_kelas_5fold.json"), "w") as f:
        json.dump(ringkasan_semua_fold, f, indent=4)

    fig_all.suptitle("Distribusi Dataset per Kelas (Fold 1 - 5)", fontsize=13, y=1.02)
    fig_all.tight_layout()
    fig_all.savefig(os.path.join(eda_dir, "distribusi_kelas_semua_fold.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_all)

    print(f"\nSemua grafik Fold 1-{k_folds} dan JSON tersimpan di folder: {eda_dir}/")


if __name__ == "__main__":
    cek_distribusi_kelas(save_dir="Results", k_folds=5)