import cv2 as cv
import torch
import numpy as np
from os import listdir
from torch.utils.data import Dataset


class getImageLabel(Dataset):
    def __init__(self, fold=1, mode='train'):
        self.dataset = []

        # Train = Original + Augmented, Valid/Test = Original saja
        orig = "Original Images/Original Images/FOLDS/fold" + str(fold)
        aug = "Augmented Images/Augmented Images/FOLDS_AUG/fold" + str(fold) + "_AUG"
        if mode == 'train':
            folders = [orig + "/Train", aug + "/Train"]
        elif mode == 'valid':
            folders = [orig + "/Valid"]
        else:
            folders = [orig + "/Test"]

        self.classes = sorted(listdir(orig + "/Train"))  # 6 kelas, urut abjad

        for folder in folders:
            for i, name in enumerate(self.classes):
                for image_name in listdir(folder + "/" + name):
                    self.dataset.append([folder + "/" + name + "/" + image_name, i])

    def __getitem__(self, item):
        path, label = self.dataset[item]
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Pre-processing CLAHE untuk memperjelas tekstur lesi kulit
        lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        l, a, b = cv.split(lab)
        l = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img = cv.cvtColor(cv.merge((l, a, b)), cv.COLOR_LAB2RGB)

        image = cv.resize(img, (128, 128)) / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)