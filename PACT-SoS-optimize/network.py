from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, concatenate, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization, Cropping2D, LeakyReLU
from tensorflow.keras import initializers, activations
import layer_object


def UNet(
        img_shape=(256, 256, 8),
        out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', dropout=0.5,
        batchnorm=False, maxpool=True, upconv=True, residual=False, padding='same'):

    def conv_block(m, dim, acti, bn, res, do, pd='same'):
        n = Conv2D(dim, 3, padding=pd, kernel_initializer=initializers.HeNormal())(m)
        n = BatchNormalization()(n) if bn else n
        n = LeakyReLU()(n) if acti == 'relu' else activations.sigmoid(n)
        n = Dropout(do)(n) if do else n
        n = Conv2D(dim, 3, padding=pd, kernel_initializer=initializers.HeNormal())(n)
        n = BatchNormalization()(n) if bn else n
        n = LeakyReLU()(n) if acti == 'relu' else activations.sigmoid(n)
        return Concatenate()(
            [Cropping2D(cropping=2 if padding == 'valid' else 0)(m),
             n]) if res else n

    def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res, pd):
        if depth > 0:
            n = conv_block(m, dim, acti, bn, res, do, pd)
            m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding=pd)(n)
            m = level_block(m, int(inc * dim), depth - 1, inc,
                            acti, do, bn, mp, up, res, pd)
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, padding='same',
                           kernel_initializer=initializers.HeNormal())(m)
                m = BatchNormalization()(m) if bn else m
                m = LeakyReLU()(m) if acti == 'relu' else activations.sigmoid(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, padding='same',
                                    kernel_initializer=initializers.HeNormal())(m)
                m = BatchNormalization()(m) if bn else m
                m = LeakyReLU()(m) if acti == 'relu' else activations.sigmoid(m)
            n = Concatenate()(
                [Cropping2D(cropping=(2 ** (depth - 1) * 12 - 8))(n),
                 m]) if padding == 'valid' else Concatenate()(
                [n, m])
            m = conv_block(n, dim, acti, bn, res, do, pd)
        else:
            m = conv_block(m, dim, acti, bn, res, do, pd)
        return m

    input = Input(shape=img_shape)

    output = level_block(input, start_ch, depth, inc_rate, activation,
                         dropout, batchnorm, maxpool, upconv, residual, padding)
    output = Conv2D(out_ch, 1, activation='tanh')(output)
    return Model(inputs=input, outputs=output)


def SegNet(input_size=(1024, 768, 8), out_ch=1, start_ch=64, height=4, inc_rate=2.,
           pool_size=(2, 2), activation='relu', dropout=0.5, batchnorm=True, residual=False):
    def conv_block(input, dim, conv_times, activation, dropout, batchnorm, residual):
        conv = input
        for i in range(conv_times):
            conv = Conv2D(dim, 3, activation=activation, padding='same')(conv)
            conv = BatchNormalization()(conv) if batchnorm else conv
        return conv if not residual else concatenate()([input, conv])

    def level_block(level, dim, depth, height, inc_rate, pool_size,
                    activation, dropout, batchnorm, residual):
        if depth < height:
            conv_times = 2 if depth < 2 else 3
            conv = conv_block(level, dim, conv_times,
                              activation, dropout, batchnorm, residual)
            mp, mp_index = layer_object.MaxPool2DWithArgmax(pool_size)(conv)
            level = level_block(mp, int(dim * inc_rate),
                                depth + 1, height, inc_rate, pool_size, activation,
                                dropout, batchnorm, residual)
            up = layer_object.MaxUnpooling2D(pool_size)([level, mp_index])
            level = conv_block(up, dim, conv_times - 1, activation,
                               dropout, batchnorm, residual)
            level = Conv2D(int(dim / inc_rate), 3,
                           activation=activation, padding='same')(level)
            level = BatchNormalization()(level) if batchnorm else level

        return level

    input = Input(shape=input_size)
    output = level_block(input, start_ch, 0, height,
                         inc_rate, pool_size, activation, 0, batchnorm, residual)
    output = Conv2D(out_ch, 1, activation='tanh')(output)
    return Model(inputs=input, outputs=output)


def SegUNet(input_size=(1024, 768, 8), out_ch=1, start_ch=64, height=4,
            inc_rate=2., pool_size=(2, 2), activation='relu', dropout=0.5,
            batchnorm=True, residual=False):
    def conv_block(input, dim, conv_times, activation, dropout, batchnorm, residual):
        conv = input
        for i in range(conv_times):
            conv = Conv2D(dim, 3, activation=activation, padding='same')(conv)
            conv = BatchNormalization()(conv) if batchnorm else conv
            conv = Dropout(dropout)(conv) if dropout else conv
        return conv if not residual else concatenate()([input, conv])

    def level_block(level, dim, depth, height, inc_rate, pool_size,
                    activation, dropout, batchnorm, residual):
        if depth < height:
            conv_times = 2 if depth < 2 else 3
            conv = conv_block(level, dim, conv_times,
                              activation, dropout, batchnorm, residual)
            mp, mp_index = layer_object.MaxPool2DWithArgmax(pool_size)(conv)
            level = level_block(mp, int(dim * inc_rate), depth + 1, height, inc_rate,
                                pool_size, activation, dropout, batchnorm, residual)
            up = layer_object.MaxUnpooling2D(pool_size)([level, mp_index])
            level = conv_block(up, dim, 1, activation,
                               dropout, batchnorm, residual)
            level = Concatenate()([conv, level])
            level = Conv2D(dim, 3, padding='same', activation=activation)(level)
            level = BatchNormalization()(level) if batchnorm else level
            level = Conv2D(int(dim / inc_rate), 3, padding='same',
                           activation=activation)(level)
            level = BatchNormalization()(level) if batchnorm else level

        return level

    input = Input(shape=input_size)
    output = level_block(input, start_ch, 0, height,
                         inc_rate, pool_size, activation, 0, batchnorm, residual)
    output = Conv2D(out_ch, 1, activation='tanh')(output)
    return Model(inputs=input, outputs=output)
