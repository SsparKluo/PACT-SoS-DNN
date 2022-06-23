import tensorflow as tf
import data_io
import cv2
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = tf.keras.models.load_model("../BME6008 - Dissertation/saved_model1")
model.summary()

x_test, y_test = data_io.load_data_test()

y = model.predict(x_test)

for i in range(2):
    imgs = np.hstack([y[i],y_test[i]])
    cv2.imshow("imgs", imgs)
    cv2.waitKey(0)

