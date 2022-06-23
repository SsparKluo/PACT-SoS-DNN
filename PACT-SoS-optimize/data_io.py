from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import random
import math
import glob


class ImageGenerator(Sequence):
    def __init__(
            self, path="../simulation/", batch_size=16, shuffle=True, mode='train',
            randon_sampling=0, rotation=True, transpose=True):
        self.path = path
        self.mode = mode
        self.random = randon_sampling
        self.rotation = rotation
        self.transpose = transpose
        self.mag = 1
        if self.mode == "train":
            self.paths = glob.glob(self.path + "[0-9]*")
        elif self.mode == 'valid':
            self.paths = glob.glob(self.path + "v[0-9]*")
        elif self.mode == 'test':
            self.paths = glob.glob(self.path + "t[0-9]*")
        if self.rotation == True:
            self.mag *= 4
        if self.transpose == True:
            self.mag *= 2
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.map = np.arange(len(self.paths) * 12 * self.mag)
        self.len = len(self.map) + self.random
        if self.shuffle == True:
            random.shuffle(self.map)
        self.operator = {
            0: lambda image: image,
            1: lambda image: cv2.flip(image, -1),
            2: lambda image: cv2.flip(image, 0),
            3: lambda image: cv2.flip(image, 1),
            4: lambda image: cv2.transpose(image),
            5: lambda image: cv2.flip(cv2.transpose(image), -1),
            6: lambda image: cv2.flip(cv2.transpose(image), 0),
            7: lambda image: cv2.flip(cv2.transpose(image), 1)
        }

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.map)

    def __len__(self):
        return math.ceil(self.len / self.batch_size)

    def __random_sample(self, sampling_num=0):
        inputs = []
        targets = []
        for i in range(sampling_num):
            file = random.randrange(len(self.paths))
            x = random.randrange(1024 - 256)
            y = random.randrange(768 - 256)
            op = random.randrange(8)
            input_filenames = glob.glob(self.paths[file] + "/SoS*.png")
            target_filenames = glob.glob(self.paths[file] + "/GT*.png")
            inputs.append(
                [self.__image_operation(
                    cv2.imread(filename, 0)[x: x + 256, y: y + 256],
                    op) for filename in input_filenames])
            targets.append(
                [self.__image_operation(
                    cv2.imread(target_filenames[0], 0)[x: x + 256, y: y + 256],
                    op)])
        return inputs, targets

    def __image_operation(self, image, operator_num=0):
        fun = self.operator.get(operator_num)
        return fun(image)

    def __getitem__(self, idx):
        if (idx + 1) * self.batch_size <= self.len - self.random:
            batch_paths = self.map[idx * self.batch_size: (idx + 1) * self.batch_size]
            inputs = []
            targets = []
            for index in batch_paths:
                input_filenames = glob.glob(
                    self.paths[index // (12 * self.mag)] + "/SoS*.png")
                target_filenames = glob.glob(
                    self.paths[index // (12 * self.mag)] + "/GT*.png")
                y = (index % (12 * self.mag) // (4 * self.mag)) * 256
                x = (index % (12 * self.mag) % (4 * self.mag) // self.mag) * 256
                op = index % (12 * self.mag) % (4 * self.mag) % self.mag
                inputs.append(
                    [self.__image_operation(
                        cv2.imread(filename, 0)[x: x + 256, y: y + 256],
                        op) for filename in input_filenames])
                targets.append(
                    [self.__image_operation(
                        cv2.imread(target_filenames[0], 0)[x: x + 256, y: y + 256],
                        op)])
            inputs = np.moveaxis((np.array(inputs) - 127.5) / 127.5, 1, 3)
            targets = np.moveaxis((np.array(targets) - 127.5) / 127.5, 1, 3)
            return inputs, targets
        elif idx * self.batch_size < self.len - self.random:
            batch_paths = self.map[idx * self.batch_size: self.len - self.random]
            inputs = []
            targets = []
            for index in batch_paths:
                input_filenames = glob.glob(
                    self.paths[index // (12 * self.mag)] + "/SoS*.png")
                target_filenames = glob.glob(
                    self.paths[index // (12 * self.mag)] + "/GT*.png")
                y = (index % (12 * self.mag) // (4 * self.mag)) * 256
                x = (index % (12 * self.mag) % (4 * self.mag) // self.mag) * 256
                op = index % (12 * self.mag) % (4 * self.mag) % self.mag
                inputs.append(
                    [self.__image_operation(
                        cv2.imread(filename, 0)[x: x + 256, y: y + 256],
                        op) for filename in input_filenames])
                targets.append(
                    [self.__image_operation(
                        cv2.imread(target_filenames[0], 0)[x: x + 256, y: y + 256],
                        op)])
            inputs = np.array(inputs)
            targets = np.array(targets)
            random_inputs, random_targets = self.__random_sample(
                (idx + 1) * self.batch_size - self.len + self.random)
            inputs = np.concatenate((inputs, random_inputs), axis=0)
            targets = np.concatenate((targets, random_targets), axis=0)
            inputs = np.moveaxis((np.array(inputs) - 127.5) / 127.5, 1, 3)
            targets = np.moveaxis((np.array(targets) - 127.5) / 127.5, 1, 3)
            return inputs, targets
        else:
            random_inputs, random_targets = self.__random_sample(self.batch_size if (
                idx + 1) * self.batch_size <= self.len else self.len - idx * self.batch_size)
            inputs = np.moveaxis((np.array(random_inputs) - 127.5) / 127.5, 1, 3)
            targets = np.moveaxis((np.array(random_targets) - 127.5) / 127.5, 1, 3)
            return inputs, targets


class ImageLoader(Sequence):
    def __init__(self, path="./splitted_data", batch_size=32, shuffle=True, mode='train'):
        self.mode = mode
        if self.mode == 'train':
            self.path = path
        elif self.mode == 'valid':
            self.path = path + '_v'
        elif self.mode == 'test':
            self.path = path + '_t'
        self.bs = batch_size
        self.shuffle = shuffle
        self.paths_y = glob.glob('{}/[0-7]/[0-9]*/y/*'.format(self.path))
        self.len = len(self.paths_y)
        self.weights = [0] * 8
        if self.mode == 'train':
            for i in range(8):
                self.weights[i] = self.len / len(
                    glob.glob('{}/{}/[0-9]*/y/*'.format(self.path, i)))
        if self.shuffle == True:
            random.shuffle(self.paths_y)

    def __len__(self):
        return math.ceil(self.len / self.bs)

    def __getitem__(self, idx):
        batch_paths_y = self.paths_y[idx * self.bs: (idx + 1) * self.bs]
        batch_paths_x = [path.replace('y', 'x') for path in batch_paths_y]
        batch_paths_x = [path[0:-4] + '(*)' + path[-4:] for path in batch_paths_x]
        batch_paths_x = [glob.glob(path) for path in batch_paths_x]
        inputs = [[cv2.imread(path, 0) for path in imgset] for imgset in batch_paths_x]
        targets = [[cv2.imread(path, 0) for path in batch_paths_y]]
        inputs = np.moveaxis((np.array(inputs) - 127.5) / 127.5, 1, 3)
        targets = np.moveaxis((np.array(targets) - 127.5) / 127.5, 0, 3)
        if self.mode == 'train':
            weights = np.array([
                self.weights[int(path[len(self.path) + 1: len(self.path) + 2])]
                for path in batch_paths_y])
            return inputs, targets, weights
        else:
            return inputs, targets

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.paths_y)
