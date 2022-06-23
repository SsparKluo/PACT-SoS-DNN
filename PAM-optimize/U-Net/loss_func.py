import tensorflow as tf
import numpy as np


def _tf_fspecial_gauss(size, sigma):
    """
    Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_gauss_conv(img, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img.get_shape().as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0
    window = _tf_fspecial_gauss(size, sigma, ch)  # window shape [size, size]
    padded_img = tf.pad(img, [[0, 0], [size//2, size//2],
                        [size//2, size//2], [0, 0]], mode="CONSTANT")
    return tf.nn.conv2d(padded_img, window, strides=[1, 1, 1, 1], padding='VALID')


def tf_gauss_weighted_l1(img1, img2, mean_metric=True, filter_size=11, filter_sigma=1.5):
    diff = tf.abs(img1 - img2)
    L1 = tf_gauss_conv(diff, filter_size=filter_size, filter_sigma=filter_sigma)
    if mean_metric:
        return tf.reduce_mean(L1)
    else:
        return L1


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img1.get_shape().as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0

    window = _tf_fspecial_gauss(size, sigma, ch)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    # 求取滑块内均值Ux Uy，均方值Ux_sq
    padded_img1 = tf.pad(img1, [[0, 0], [size//2, size//2],
                         [size//2, size//2], [0, 0]], mode="CONSTANT")  # img1 上下左右补零
    padded_img2 = tf.pad(img2, [[0, 0], [size//2, size//2],
                         [size//2, size//2], [0, 0]], mode="CONSTANT")  # img2 上下左右补零
    mu1 = tf.nn.conv2d(padded_img1, window, strides=[
                       1, 1, 1, 1], padding='VALID')  # 利用滑动窗口，求取窗口内图像的的加权平均
    mu2 = tf.nn.conv2d(padded_img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1*mu1  # img(x,y) Ux*Ux 均方
    mu2_sq = mu2*mu2  # img(x,y) Uy*Uy
    mu1_mu2 = mu1*mu2  # img(x,y) Ux*Uy

    # 求取方差，方差等于平方的期望减去期望的平方，平方的均值减去均值的平方
    paddedimg11 = padded_img1*padded_img1
    paddedimg22 = padded_img2*padded_img2
    paddedimg12 = padded_img1*padded_img2

    sigma1_sq = tf.nn.conv2d(paddedimg11, window, strides=[
                             1, 1, 1, 1], padding='VALID') - mu1_sq  # sigma1方差
    sigma2_sq = tf.nn.conv2d(paddedimg22, window, strides=[
                             1, 1, 1, 1], padding='VALID') - mu2_sq  # sigma2方差
    sigma12 = tf.nn.conv2d(paddedimg12, window, strides=[
                           1, 1, 1, 1], padding='VALID') - mu1_mu2  # sigma12协方差，乘积的均值减去均值的乘积

    ssim_value = tf.clip_by_value(((2*mu1_mu2 + C1)*(2*sigma12 + C2)) /
                                  ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)), 0, 1)
    if cs_map:  # 只考虑contrast对比度，structure结构，不考虑light亮度
        cs_map_value = tf.clip_by_value(
            (2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2), 0, 1)  # 对比度结构map
        value = (ssim_value, cs_map_value)
    else:
        value = ssim_value
    if mean_metric:  # 求取矩阵的均值，否则返回ssim矩阵
        value = tf.reduce_mean(value)
    return value


def tf_ssim_l1_loss(img1, img2, mean_metric=True, filter_size=11, filter_sigma=1.5, alpha=0.84):
    L1 = tf_gauss_weighted_l1(img1, img2, mean_metric=False,
                              filter_size=filter_size, filter_sigma=filter_sigma)
    if mean_metric:
        loss_ssim = 1 - tf_ssim(img1, img2, cs_map=False, mean_metric=True,
                                filter_size=filter_size, filter_sigma=filter_sigma)
        loss_L1 = tf.reduce_mean(L1)
        value = loss_ssim * alpha + loss_L1 * (1-alpha)
    else:
        loss_ssim = 1 - tf_ssim(img1, img2, cs_map=False, mean_metric=False,
                                filter_size=filter_size, filter_sigma=filter_sigma)
        value = loss_ssim * alpha + L1 * (1-alpha)

    return value, loss_ssim
