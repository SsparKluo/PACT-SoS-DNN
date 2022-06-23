import cv2
import numpy as np
import random

from scipy import rand


def load_data_train():
    X_train = []
    Y_train = []

    fp_y_train1 = "./PAM-optimize/splitted_data/11/"
    fp_y_train2 = "./PAM-optimize/splitted_data/16/"
    fp_x_train1 = "./PAM-optimize/splitted_data/1/"
    fp_x_train2 = "./PAM-optimize/splitted_data/8/"

    for i in range(90):
        for j in range(8):
            img1 = cv2.imread(fp_x_train1+str(i)+str(j)+".png", 0)
            img2 = cv2.imread(fp_x_train2+str(i)+str(j)+".png", 0)
            ref1 = cv2.imread(fp_y_train1+str(i)+str(j)+".png", 0)
            ref2 = cv2.imread(fp_y_train2+str(i)+str(j)+".png", 0)

            X_train.append(np.array([x/255.0 for x in img1]).reshape(256, 256, 1))
            X_train.append(np.array([x/255.0 for x in img2]).reshape(256, 256, 1))
            Y_train.append(np.array([x/255.0 for x in ref1]).reshape(256, 256, 1))
            Y_train.append(np.array([x/255.0 for x in ref2]).reshape(256, 256, 1))

    return np.array(X_train), np.array(Y_train)


def load_data_test():
    X_test = []
    Y_test = []

    fp_y_test1 = "./PAM-optimize/splitted_data/11/"
    fp_y_test2 = "./PAM-optimize/splitted_data/16/"
    fp_x_test1 = "./PAM-optimize/splitted_data/5/"
    fp_x_test2 = "./PAM-optimize/splitted_data/10/"

    for i in range(30):
        img1 = cv2.imread(fp_x_test1+str(i)+str(0)+".png", 0)
        img2 = cv2.imread(fp_x_test2+str(i)+str(0)+".png", 0)
        ref1 = cv2.imread(fp_y_test1+str(i)+str(0)+".png", 0)
        ref2 = cv2.imread(fp_y_test2+str(i)+str(0)+".png", 0)

        X_test.append(np.array([x/255.0 for x in img1]).reshape(256, 256, 1))
        X_test.append(np.array([x/255.0 for x in img2]).reshape(256, 256, 1))
        Y_test.append(np.array([x/255.0 for x in ref1]).reshape(256, 256, 1))
        Y_test.append(np.array([x/255.0 for x in ref2]).reshape(256, 256, 1))

    return np.array(X_test), np.array(Y_test)


def train_data_generator(bs):
    path = "./PAM-optimize/splitted_data/"
    while True:
        indexes = []
        x_train = []
        y_train = []
        for f in [1, 2, 3, 6, 7, 8]:
            for x in range(90):
                for r in range(8):
                    indexes.append(str(f)+str(x)+str(r))
        indexes = np.array(indexes)
        a = 0
        while a < 4320:
            choiced_indexes = []
            if len(indexes) >= bs:
                choiced_indexes = np.random.choice(indexes.shape[0], bs, replace=False)
            else:
                choiced_indexes = np.random.choice(
                    indexes.shape[0], len(indexes), replace=False)
            original_indexes = np.arange(indexes.shape[0])
            left_indexes = np.delete(original_indexes, choiced_indexes)
            indexes = indexes[left_indexes]
            for j in choiced_indexes:
                i = indexes[j]
                img = cv2.imread(path+i[0]+"/"+i[1:-1]+i[-1]+".png", 0)
                ref = cv2.imread(path+str(11)+"/"+i[1:-1]+i[-1]+".png", 0) if i[0] in [
                    '1', '2', '3'] else cv2.imread(path+str(16)+"/"+i[1:-1]+i[-1]+".png", 0)
                x_train.append(np.array([x/255.0 for x in img]).reshape(256, 256, 1))
                y_train.append(np.array([x/255.0 for x in ref]).reshape(256, 256, 1))
            yield (np.array(x_train), np.array(y_train))


def test_data_generator(bs):
    path = "./PAM-optimize/splitted_data/"
    while True:
        x_test = []
        y_test = []
        file = random.choices([5, 10], k=bs)
        y = random.choices([j for j in range(8)], k=bs)
        for i in range(bs):
            x = random.randrange(90)
            img = cv2.imread(path+str(file[i])+"/"+str(x)+str(y[i])+".png", 0)
            ref = cv2.imread(path+str(file[i]+10)+"/"+str(x)+str(y[i])+".png", 0)
            x_test.append(np.array([x/255.0 for x in img]).reshape(256, 256, 1))
            y_test.append(np.array([x/255.0 for x in ref]).reshape(256, 256, 1))
        yield (np.array(x_test), np.array(y_test))
