import data_io
import network
import tensorflow as tf
import cv2
import os
import numpy as np
from glob import glob
from loss_func import regression_loss


def image_saved(input, model='unet'):
    batchs = [input[x:x + 12] for x in range(0, len(input), 12)]
    result = [batch.reshape(4, 3, 256, 256, 1)
              .reshape(1024,768) * 255 for batch in batchs]
    folder = './figure/test_result_' + model
    if not os.path.exists(folder):
        os.mkdir(folder)
    for idx, img in enumerate(result):
        filename = folder + "/" + str(idx) + '.png'
        cv2.imwrite(filename, img)


if __name__ == "__main__":
    restart = True
    while(restart):
        mode = input("Which network to be evaluated? (unet/segnet/segunet): ")
        if mode == 'unet' or mode == 'UNet':
            print("Evaluating U-Net:")
            restart = False
            model_file = "./saved_model/unet"
        if mode == 'SegNet' or mode == 'segnet':
            print("Evaluating SegNet:")
            restart = False
            model_file = "./saved_model/segnet"
        if mode == 'segunet' or mode == 'SegUNet':
            print("Evaluating SegUNet:")
            restart = False
            model_file = "./saved_model/segunet"
        model = tf.keras.models.load_model(
            model_file, custom_objects={'regression_loss': regression_loss})
        test_loader = data_io.SequenceData(
            batch_size=12,
            shuffle=False,
            mode='test',
            randon_sampling=0,
            rotation=False,
            transpose=False)
        y = model.predict(test_loader)
        image_saved(y, mode)
