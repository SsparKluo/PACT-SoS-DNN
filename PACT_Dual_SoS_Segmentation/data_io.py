from scipy.io import loadmat
from glob import glob
from tensorflow.keras.utils import Sequence
from math import ceil

import numpy as np
import cv2
import random


class ImageDataGenerator(Sequence):
    def __init__(self, path="../simulation_dual_SoS/", batch_size=8,
                 shuffle=True, mode="train", data_augmentation="True", output_size=(192,256)) -> None:
        super().__init__()
        self.mode = mode
        if self.mode == 'train':
            self.paths = glob("{}[0-9]*".format(path))
        elif self.mode == 'valid':
            self.paths = glob("{}v_[0-9]*".format(path))
        elif self.mode == 'test':
            self.paths = glob("{}t_[0-9]*".format(path))

        self.bs = batch_size
        self.shuffle = shuffle
        self.aug = data_augmentation
        self.mag = 4 if self.aug else 2
        self.len = len(self.paths) * self.mag
        self.indexes = np.arange(self.len)
        self.output_size = output_size
        if self.shuffle:
            random.shuffle(self.indexes)

    def __len__(self):
        return ceil(self.len / self.bs)

    def __getitem__(self, idx):
        batch_paths = self.indexes[idx * self.bs: (idx + 1) * self.bs]
        inputs = []
        targets = []
        for i in batch_paths:
            folderpath = self.paths[i // self.mag]
            file_idx = i % self.mag // 2
            # 0 for 128 channels image; 1 for 64 channels image

            img = cv2.imread("{}/input_128.png".format(folderpath), 0) \
                if file_idx == 0 else \
                cv2.imread("{}/input_64.png".format(folderpath), 0)

            img = cv2.flip(img, 1) if self.aug and i % 2 else img
            # 0 for raw image; 1 for flipping
            img = cv2.resize(img, self.output_size)
            gt = loadmat("{}/GT.mat".format(folderpath))
            gt = gt['GT']
            gt = cv2.flip(gt, 1) if self.aug and i % 2 else gt
            gt = (gt - gt.min()) / (gt.max() - gt.min())
            gt = cv2.resize(gt, self.output_size)
            inputs.append(np.array(img))
            targets.append(np.array(gt))
        return np.expand_dims(
            np.array(inputs), axis=-1), np.expand_dims(np.array(targets), axis=-1)

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.indexes)


class RawDataGenerator(Sequence):
    pass
