from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization, Cropping2D, LeakyReLU
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras import initializers, activations
from keras.backend import int_shape
# import layer_object


def unet(
        img_shape=(512, 384, 1),
        out_ch=1, start_ch=64, depth=3, inc_rate=2., activation='relu', dropout=0.5,
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
    output = Conv2D(out_ch, 1, activation='sigmoid')(output)
    return Model(inputs=input, outputs=output)


def unet_with_denses(img_shape=(512, 384, 1),
                     out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
                     dropout=0.5, batchnorm=False, maxpool=True, upconv=True,
                     residual=False, padding='same'):

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

    def dense_link(m, acti, do, depth):
        shape = (32 * (depth + 1), 24 * (depth + 1))
        n = Flatten()(m)
        n = Dense(units=24 * (depth + 1), kernel_initializer=initializers.HeNormal())(n)
        n = LeakyReLU()(m) if acti == 'relu' else activations.sigmoid(m)
        n = Dropout(do)(n) if do else n
        n = Dense(units=shape, kernel_initializer=initializers.HeNormal())(n)
        return Reshape(target_shape=(shape[0], shape[1], 1))(n)

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
            n = dense_link(n, acti, do, depth) if depth < 2 else n
            n = Concatenate()(
                [Cropping2D(cropping=(2 ** (depth - 1) * 12 - 8))(n),
                 m]) if padding == 'valid' else Concatenate()(
                [n, m]) if depth < 2 else m
            m = conv_block(n, dim, acti, bn, res, do, pd)
        else:
            m = conv_block(m, dim, acti, bn, res, do, pd)
        return m

    input = Input(shape=img_shape)

    output = level_block(input, start_ch, depth, inc_rate, activation,
                         dropout, batchnorm, maxpool, upconv, residual, padding)
    output = Conv2D(out_ch, 1, activation='sigmoid')(output)
    return Model(inputs=input, outputs=output)
