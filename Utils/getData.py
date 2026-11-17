import cv2 as cv
import torch
import numpy as np
from os import listdir
from torch.utils.data import Dataset


class getImageLabel(Dataset):
    def __init__(self, fold=1, mode='train'):
        self.dataset = []
        self.mode = mode

        orig = "Original Images/Original Images/FOLDS/fold" + str(fold)
        aug = "Augmented Images/Augmented Images/FOLDS_AUG/fold" + str(fold) + "_AUG"
        if mode == 'train':
            folders = [orig + "/Train", aug + "/Train"]
        elif mode == 'valid':
            folders = [orig + "/Valid"]
        else:
            folders = [orig + "/Test"]

        self.classes = sorted(listdir(orig + "/Train"))  

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

        
        if self.mode == 'train':
            if np.random.rand() > 0.5:
                img = cv.flip(img, 1)  
            if np.random.rand() > 0.5:
                img = cv.flip(img, 0)  
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k).copy()  

        # Resize & normalisasi standar RGB agar gradien lebih stabil
        image = cv.resize(img, (128, 128)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)