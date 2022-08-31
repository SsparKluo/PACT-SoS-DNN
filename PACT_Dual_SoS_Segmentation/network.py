from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.layers import Conv1D, UpSampling1D
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization, Cropping2D, LeakyReLU
from tensorflow.keras.layers import Dense, Flatten, Reshape
#from tensorflow.keras.layers import ConvLSTM1D, TimeDistributed, RepeatVector
from tensorflow.keras import initializers, activations
from keras.backend import int_shape
# import layer_object


def unet(
        img_shape=(512, 384, 1),
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
    output = Conv2D(out_ch, 1, activation='sigmoid')(output)
    return Model(inputs=input, outputs=output)


def unet_with_dense(img_shape=(512, 384, 1),
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
        shape = (16 * 2**depth, 12 * 2**depth)
        n = Flatten()(m)
        n = Dense(units=12 * 2**depth,
                  kernel_initializer=initializers.HeNormal())(n)
        n = BatchNormalization()(n)
        n = LeakyReLU()(n) if acti == 'relu' else activations.sigmoid(n)
        n = Dropout(do)(n) if do else n
        n = Dense(units=shape[0] * shape[1],
                  kernel_initializer=initializers.HeNormal())(n)
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
            n = dense_link(n, acti, do, depth) if depth <= 2 else n
            n = Concatenate()(
                [Cropping2D(cropping=(2 ** (depth - 1) * 12 - 8))(n),
                 m]) if padding == 'valid' else Concatenate()(
                [n, m]) if depth <= 2 else m
            m = conv_block(n, dim, acti, bn, res, do, pd)
        else:
            m = conv_block(m, dim, acti, bn, res, do, pd)
            m = dense_link(m, acti, do, depth)
        return m

    input = Input(shape=img_shape)

    output = level_block(input, start_ch, depth, inc_rate, activation,
                         dropout, batchnorm, maxpool, upconv, residual, padding)
    output = Conv2D(out_ch, 1, activation='sigmoid')(output)
    return Model(inputs=input, outputs=output)


def fcn_dense(img_shape=(256, 192, 1),
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

    def dense_link(m, acti, do, inner_dim, out_dim):
        n = Flatten()(m)
        n = Dense(units=inner_dim, kernel_initializer=initializers.HeNormal())(n)
        n = BatchNormalization()(n)
        n = LeakyReLU()(n) if acti == 'relu' else activations.sigmoid(n)
        n = Dropout(do)(n) if do else n
        n = Dense(units=out_dim, kernel_initializer=initializers.HeNormal())(n)
        return Reshape((out_dim, 1))(n)

    input = Input(shape=img_shape)

    conv1 = conv_block(input, start_ch, activation,
                       batchnorm, residual, dropout, padding)
    maxpool1 = MaxPooling2D()(conv1)

    conv2 = conv_block(maxpool1, start_ch * 2, activation,
                       batchnorm, residual, dropout, padding)
    maxpool2 = MaxPooling2D()(conv2)
    dense2 = dense_link(maxpool2, activation, dropout, 192, 96)

    conv3 = conv_block(maxpool2, start_ch * 4, activation,
                       batchnorm, residual, dropout, padding)
    maxpool3 = MaxPooling2D()(conv3)
    dense3 = dense_link(maxpool3, activation, dropout, 192, 48)

    conv4 = conv_block(maxpool3, start_ch * 8, activation,
                       batchnorm, residual, dropout, padding)
    maxpool4 = MaxPooling2D()(conv4)
    dense4 = dense_link(maxpool4, activation, dropout, 192, 24)

    up_dense4 = UpSampling1D(size=2)(dense4)
    concat3 = Concatenate(axis=2)([up_dense4, dense3])
    concat3 = Conv1D(4, 3, padding=padding,
                     kernel_initializer=initializers.HeNormal())(concat3)
    concat3 = BatchNormalization()(concat3)
    concat3 = LeakyReLU()(concat3)
    concat3 = Conv1D(2, 3, padding=padding,
                     kernel_initializer=initializers.HeNormal())(concat3)
    concat3 = BatchNormalization()(concat3)
    concat3 = LeakyReLU()(concat3)

    up_dense3 = UpSampling1D(size=2)(concat3)
    # concat2 = Concatenate(axis=2)([up_dense3, dense2])
    concat2 = Conv1D(4, 3, padding=padding,
                     kernel_initializer=initializers.HeNormal())(up_dense3)
    concat2 = BatchNormalization()(concat2)
    concat2 = LeakyReLU()(concat2)
    concat2 = Conv1D(2, 3, padding=padding,
                     kernel_initializer=initializers.HeNormal())(concat2)
    concat2 = BatchNormalization()(concat2)
    concat2 = LeakyReLU()(concat2)

    up_dense2 = UpSampling1D(size=2)(concat2)
    concat1 = Conv1D(4, 3, padding=padding,
                     kernel_initializer=initializers.HeNormal())(up_dense2)
    concat1 = BatchNormalization()(concat1)
    concat1 = LeakyReLU()(concat1)
    concat1 = Conv1D(2, 3, padding=padding,
                     kernel_initializer=initializers.HeNormal())(concat1)
    concat1 = BatchNormalization()(concat1)
    concat1 = LeakyReLU()(concat1)

    output = Conv1D(out_ch, 1)(concat1)
    return Model(inputs=input, outputs=output)


def cnn_dense(img_shape=(256, 192, 1),
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
        n = Flatten()(m)
        n = Dense(units=32 * 4**depth, kernel_initializer=initializers.HeNormal())(n)
        n = BatchNormalization()(n)
        n = LeakyReLU()(n) if acti == 'relu' else activations.sigmoid(n)
        n = Dropout(do)(n) if do else n
        return n

    input = Input(shape=img_shape)
    n = conv_block(input, start_ch, activation, batchnorm, residual, dropout, padding)
    n = MaxPooling2D()(n)
    n = conv_block(n, 2 * start_ch, activation, batchnorm, residual, dropout, padding)
    n = MaxPooling2D()(n)
    n = conv_block(n, 4 * start_ch, activation, batchnorm, residual, dropout, padding)
    n = MaxPooling2D()(n)
    m1 = conv_block(n, 8 * start_ch, activation, batchnorm, residual, dropout, padding)
    n = MaxPooling2D()(m1)
    m2 = conv_block(n, 16 * start_ch, activation, batchnorm, residual, dropout, padding)
    n = MaxPooling2D()(m2)
    m3 = conv_block(n, 16 * start_ch, activation, batchnorm, residual, dropout, padding)
    dense3 = dense_link(m3, activation, dropout, 0)
    dense2 = dense_link(m2, activation, dropout, 0)
    dense1 = dense_link(m1, activation, dropout, 0)
    n = Concatenate()([dense3, dense2, dense1])
    n = Dense(units=640, kernel_initializer=initializers.HeNormal())(n)
    n = BatchNormalization()(n)
    n = LeakyReLU()(n)
    n = Dropout(dropout)(n) if dropout else n
    n = Dense(units=320, kernel_initializer=initializers.HeNormal())(n)
    n = BatchNormalization()(n)
    n = LeakyReLU()(n)
    n = Dropout(dropout)(n) if dropout else n
    n = Dense(units=160, kernel_initializer=initializers.HeNormal())(n)
    n = BatchNormalization()(n)
    n = LeakyReLU()(n)
    n = Dense(units=1, activation='linear')(n)
    output = n
    return Model(inputs=input, outputs=output)


'''
def convlstm(img_shape=(901, 1, 128, 1), out_ch=1, start_ch=8,
             activation='relu', dropout=0.5, padding='same'):

    input = Input(shape=img_shape, name='input')

    seq = ConvLSTM1D(start_ch, 5, padding=padding, data_format='channels_last',
                     activation=activation, kernel_initializer=initializers.HeNormal(),
                     return_sequences=True, dropout=dropout, recurrent_dropout=0.2)(input)
    seq = ConvLSTM1D(start_ch, 3, padding=padding, data_format='channels_last',
                     activation=activation, kernel_initializer=initializers.HeNormal(),
                     dropout=dropout, recurrent_dropout=0.2)(seq)
    decoder_input = RepeatVector(img_shape[0])
    seq = ConvLSTM1D(start_ch, 3, padding=padding, data_format='channels_last',
                     activation=activation, kernel_initializer=initializers.HeNormal(),
                     return_sequences=True, dropout=dropout,
                     recurrent_dropout=0.2)(decoder_input)
    seq = ConvLSTM1D(start_ch, 5, padding=padding, data_format='channels_last',
                     activation=activation, kernel_initializer=initializers.HeNormal(),
                     return_sequences=True, dropout=dropout, recurrent_dropout=0.2)(seq)
    output = TimeDistributed(Conv1D(out_ch, 3, padding=padding))(seq)
    return Model(inputs=input, outputs=output)
'''
