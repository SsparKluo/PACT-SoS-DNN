from cmath import pi
from turtle import xcor
import cv2
import re
from glob import glob
import os
import random
from matplotlib.pyplot import pie

from scipy import rand


def image_cut(img_path, size=(256, 256), padding=0):
    img = cv2.imread(img_path, 0)
    y_num = img.shape[0] // size[0]
    x_num = img.shape[1] // size[1]
    # Make borders
    if not padding:
        img = cv2.copyMakeBorder(img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, 0)
        img = cv2.copyMakeBorder(img, 0, 0, padding, padding, cv2.BORDER_REFLECT)
    pieces = [img[y * size[0]: (y + 1) * size[0] + 2 * padding,
                  x * size[1]: (x + 1) * size[1] + 2 * padding]
              for y in range(y_num) for x in range(x_num)]
    return pieces


def data_augmentation(img_folder, output_size=(256, 256), padding=0,
                      rescale=[1.5, 2.], rescale_num=6, flip=True):
    x_paths = glob('{}/SoS_*'.format(img_folder))
    y_paths = glob('{}/GT_*'.format(img_folder))
    x_imgs = [cv2.imread(img_path, 0) for img_path in x_paths]
    y_img = cv2.imread(y_paths[0], 0)
    img_shape = y_img.shape
    if not padding:
        x_imgs = [cv2.copyMakeBorder(img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, 0)
                  for img in x_imgs]
        x_imgs = [cv2.copyMakeBorder(img, 0, 0, padding, padding, cv2.BORDER_REFLECT)
                  for img in x_imgs]
        y_img = cv2.copyMakeBorder(y_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, 0)
        y_img = cv2.copyMakeBorder(y_img, 0, 0, padding, padding, cv2.BORDER_REFLECT)
    batch_pieces_x = []
    pieces_y = []
    if flip:
        size = (output_size[0] - padding, output_size[1] - padding)
        y_num = img_shape[0] // size[0]
        x_num = img_shape[1] // size[1]
        batch_pieces_x = [[img[y * size[0]: (y + 1) * size[0] + 2 * padding,
                               x * size[1]: (x + 1) * size[1] + 2 * padding]
                           for img in x_imgs]
                          for y in range(y_num) for x in range(x_num)]
        batch_pieces_x = [[cv2.flip(img, 1) for img in batch] for batch in batch_pieces_x]
        pieces_y = [y_img[y * size[0]: (y + 1) * size[0] + 2 * padding,
                          x * size[1]: (x + 1) * size[1] + 2 * padding]
                    for y in range(y_num) for x in range(x_num)]
    for i in range(len(rescale)):
        y_limit = (img_shape[0] + 2 * padding) - int(output_size[0] * rescale[i])
        x_limit = (img_shape[1] + 2 * padding) - int(output_size[1] * rescale[i])
        for _ in range(rescale_num):
            y = random.randrange(y_limit)
            x = random.randrange(x_limit)
            batch_pieces_x.append(
                [cv2.resize(img[y: y + int(rescale[i] * output_size[0]),
                                x: x + int(rescale[i] * output_size[1])],
                            output_size) for img in x_imgs])
            pieces_y.append(
                cv2.resize(y_img[y: y + int(rescale[i] * output_size[0]),
                                 x: x + int(rescale[i] * output_size[1])],
                           output_size))
    return batch_pieces_x, pieces_y


if __name__ == '__main__':
    # Train data x: ./splitted_data/{GT SoS class}/{img idx}/x/{position idx}({SoS idx}).png
    # Train data y: ./splitted_data/{GT SoS class}/{img idx}/y/{position idx}.png
    path = "../simulation/"
    train_paths = glob(path + "[0-9]*")
    valid_paths = glob(path + "v[0-9]*")
    test_paths = glob(path + "t[0-9]*")
    total_train = len(train_paths)
    class_num = [0] * 8
    re_gt = re.compile(r'GT_[0-9]{4}')
    mode = input("Choose a mode(train, valid, test): ")
    if mode == 'train':
        if not os.path.exists('./splitted_data/'):
            os.mkdir('./splitted_data/')
            for i in range(8):
                os.mkdir('./splitted_data/' + str(i))
        for p in train_paths:
            gt_filename = re.findall(re_gt, glob(p + '/GT*')[0])
            gt_sos = int(gt_filename[0][3:])
            sos_idx = (gt_sos - 1460) // 20
            x_files = glob(p + '/SoS_*')
            os.mkdir('./splitted_data/{}/{}'.format(str(sos_idx),
                     str(class_num[sos_idx])))
            os.mkdir('./splitted_data/{}/{}/x'.format(str(sos_idx),
                     str(class_num[sos_idx])))
            os.mkdir('./splitted_data/{}/{}/y'.format(str(sos_idx),
                     str(class_num[sos_idx])))
            for idx_file, p_x in enumerate(x_files):
                pieces = image_cut(p_x, padding=0)
                for idx_img, img in enumerate(pieces):
                    cv2.imwrite('./splitted_data/{}/{}/x/{}({}).png'.format(str(sos_idx),
                                str(class_num[sos_idx]), str(idx_img), str(idx_file)), img)
            pieces = image_cut(p + '/' + gt_filename[0] + '.png', padding=0)
            exist_img = len(pieces)
            for idx_img, img in enumerate(pieces):
                cv2.imwrite('./splitted_data/{}/{}/y/{}.png'.format(str(sos_idx),
                            str(class_num[sos_idx]), str(idx_img)), img)
            data_aug_x, data_aug_y = data_augmentation(p)
            for idx_img, img_set in enumerate(data_aug_x):
                for idx, img in enumerate(img_set):
                    cv2.imwrite('./splitted_data/{}/{}/x/{}({}).png'.
                                format(str(sos_idx), str(class_num[sos_idx]),
                                       str(exist_img + idx_img), str(idx)), img)
            for idx_img, img in enumerate(data_aug_y):
                cv2.imwrite('./splitted_data/{}/{}/y/{}.png'.format(str(sos_idx),
                            str(class_num[sos_idx]), str(exist_img + idx_img)), img)
            class_num[sos_idx] += 1

    elif mode == 'valid':
        if not os.path.exists('./splitted_data_v/'):
            os.mkdir('./splitted_data_v/')
            for i in range(8):
                os.mkdir('./splitted_data_v/' + str(i))
        for p in valid_paths:
            gt_filename = re.findall(re_gt, glob(p + '/GT*')[0])
            gt_sos = int(gt_filename[0][3:])
            sos_idx = (gt_sos - 1460) // 20
            x_files = glob(p + '/SoS_*')
            os.mkdir('./splitted_data_v/{}/{}'.format(str(sos_idx),
                     str(class_num[sos_idx])))
            os.mkdir('./splitted_data_v/{}/{}/x'.format(str(sos_idx),
                     str(class_num[sos_idx])))
            os.mkdir('./splitted_data_v/{}/{}/y'.format(str(sos_idx),
                     str(class_num[sos_idx])))
            for idx_file, p_x in enumerate(x_files):
                pieces = image_cut(p_x, padding=0)
                for idx_img, img in enumerate(pieces):
                    cv2.imwrite('./splitted_data_v/{}/{}/x/{}({}).png'.format(str(sos_idx),
                                str(class_num[sos_idx]), str(idx_img), str(idx_file)), img)
            pieces = image_cut(p + '/' + gt_filename[0] + '.png', padding=0)
            for idx_img, img in enumerate(pieces):
                cv2.imwrite('./splitted_data_v/{}/{}/y/{}.png'.format(str(sos_idx),
                            str(class_num[sos_idx]), str(idx_img)), img)
            class_num[sos_idx] += 1
    elif mode == 'test':
        if not os.path.exists('./splitted_data_t/'):
            os.mkdir('./splitted_data_t/')
            for i in range(8):
                os.mkdir('./splitted_data_t/' + str(i))
        for p in test_paths:
            gt_filename = re.findall(re_gt, glob(p + '/GT*')[0])
            gt_sos = int(gt_filename[0][3:])
            sos_idx = (gt_sos - 1460) // 20
            x_files = glob(p + '/SoS_*')
            os.mkdir('./splitted_data_t/{}/{}'.format(str(sos_idx),
                     str(class_num[sos_idx])))
            os.mkdir('./splitted_data_t/{}/{}/x'.format(str(sos_idx),
                     str(class_num[sos_idx])))
            os.mkdir('./splitted_data_t/{}/{}/y'.format(str(sos_idx),
                     str(class_num[sos_idx])))
            for idx_file, p_x in enumerate(x_files):
                pieces = image_cut(p_x, padding=0)
                for idx_img, img in enumerate(pieces):
                    cv2.imwrite('./splitted_data_t/{}/{}/x/{}({}).png'.format(str(sos_idx),
                                str(class_num[sos_idx]), str(idx_img), str(idx_file)), img)
            pieces = image_cut(p + '/' + gt_filename[0] + '.png', padding=0)
            for idx_img, img in enumerate(pieces):
                cv2.imwrite('./splitted_data_t/{}/{}/y/{}.png'.format(str(sos_idx),
                            str(class_num[sos_idx]), str(idx_img)), img)
            class_num[sos_idx] += 1
