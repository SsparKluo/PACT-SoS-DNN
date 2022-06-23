import os
import data_io
import network
import loss_func
import matplotlib.pyplot as plt
from glob import glob


def unet_train_new_v1(img_shape=(256, 256, 8), out_ch=1, start_ch=64, depth=4,
                      inc_rate=2., activation='relu', dropout=0.5, batchnorm=True,
                      maxpool=True, upconv=True, residual=False, padding='same'):
    print("Training for a new UNet model:")
    import tensorflow.keras.optimizers as optim
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.utils import plot_model

    model = network.UNet(img_shape, out_ch, start_ch, depth,
                         inc_rate, activation, dropout, batchnorm,
                         maxpool, upconv, residual, padding='same')
    # model.summary()
    # plot_model(model, to_file='model_unet.png', show_shapes=True)

    model.compile(loss=loss_func.regression_loss, optimizer=optim.Adam())

    # model.compile(loss='mean_squared_error', optimizer=optim.Adam(learning_rate=0.001))
    bs = 32
    reduce_lr = ReduceLROnPlateau(
        verbose=1, factor=0.1, min_delta=0.1, monitor='val_loss', patience=10,
        mode='auto', min_lr=0.00001)
    stop_condition = EarlyStopping(
        monitor='val_loss', patience=21, min_delta=0.1)

    trainset_loader = data_io.ImageLoader(batch_size=bs)
    #validset_loader = data_io.ImageLoader(batch_size=bs, mode='valid')
    validset_loader = data_io.ImageLoader(batch_size=bs, mode='valid', shuffle=False)

    history = model.fit(trainset_loader, batch_size=bs, epochs=100,
                        validation_data=validset_loader, workers=16, max_queue_size=16,
                        callbacks=[reduce_lr, stop_condition])

    model.save("./saved_model/unet6")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./figure/UNet - Model loss6.jpg')


def segnet_train(img_shape=(256, 256, 8), out_ch=1, start_ch=64, depth=4,
                 inc_rate=2., activation='relu', dropout=0.5, batchnorm=False,
                 maxpool=True, upconv=True, residual=False):
    print("Training for a new SegNet model:")
    import tensorflow.keras.optimizers as optim
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.utils import plot_model

    model = network.SegNet(img_shape, out_ch, start_ch, depth, inc_rate,
                           activation, dropout, batchnorm, maxpool, upconv, residual)
    # model.summary()
    plot_model(model, to_file='model_segnet.png', show_shapes=True)
    model.compile(loss=loss_func.regression_loss, optimizer=optim.Adam(
        learning_rate=0.01), run_eagerly=True)
    bs = 4
    reduce_lr = ReduceLROnPlateau(
        verbose=1, factor=0.1, min_delta=0.0001, monitor='val_loss', patience=10,
        mode='auto', min_lr=0.00001)
    stop_condition = EarlyStopping(
        monitor='val_loss', patience=10, min_delta=0.01)

    trainset_loader = data_io.SequenceData(batch_size=bs)
    validset_loader = data_io.SequenceData(batch_size=bs, mode='valid')

    history = model.fit(trainset_loader, batch_size=bs, epochs=10,
                        validation_data=validset_loader,
                        callbacks=[reduce_lr, stop_condition], use_multiprocessing=True)

    model.save("./saved_model/segnet")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./figure/SegNet - Model loss.jpg')


def segunet_train(img_shape=(256, 256, 8), out_ch=1, start_ch=64, depth=4,
                  inc_rate=2., activation='relu', dropout=0.5, batchnorm=False,
                  maxpool=True, upconv=True, residual=False):
    print("Training for a new SegNet model:")
    import tensorflow.keras.optimizers as optim
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.utils import plot_model

    model = network.SegUNet(img_shape, out_ch, start_ch, depth, inc_rate,
                            activation, dropout, batchnorm, maxpool, upconv, residual)
    # model.summary()
    plot_model(model, to_file='model_segunet.png', show_shapes=True)
    model.compile(loss=loss_func.regression_loss, optimizer=optim.Adam(
        learning_rate=0.01), run_eagerly=True)
    bs = 4
    reduce_lr = ReduceLROnPlateau(
        verbose=1, factor=0.1, min_delta=0.0001, monitor='val_loss', patience=10,
        mode='auto', min_lr=0.00001)
    stop_condition = EarlyStopping(
        monitor='val_loss', patience=10, min_delta=0.01)

    trainset_loader = data_io.SequenceData(batch_size=bs)
    validset_loader = data_io.SequenceData(batch_size=bs, mode='valid')

    history = model.fit(trainset_loader, batch_size=bs, epochs=10,
                        validation_data=validset_loader,
                        callbacks=[reduce_lr, stop_condition], use_multiprocessing=True)

    model.save("./saved_model/segunet")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./figure/SegUNet - Model loss.jpg')


def train_from_model():
    print("Train from an exist model:")
    from tensorflow.keras.models import load_model
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.optimizers import Adam

    saved_models = glob("./saved_model/*")
    print("Models found:")
    for p in saved_models:
        print(p)
    path = input("Which one to be loaded? ")
    model = load_model(path, compile=False)

    model.compile(loss=loss_func.regression_loss, optimizer=Adam(learning_rate=0.00001))

    bs = 32
    num_epoch = int(input("Continue from which epoch: "))
    trainset_loader = data_io.ImageLoader(batch_size=bs)
    validset_loader = data_io.ImageLoader(batch_size=bs, mode='valid', shuffle=False)
    '''
    reduce_lr = ReduceLROnPlateau(
        verbose=1, factor=0.1, min_delta=1, monitor='val_loss', patience=5,
        mode='auto', min_lr=0.000001)
    stop_condition = EarlyStopping(
        monitor='val_loss', patience=11, min_delta=1)
    '''

    history = model.fit(trainset_loader, batch_size=bs,
                        epochs=80, initial_epoch=num_epoch,
                        validation_data=validset_loader, workers=16,
                        max_queue_size=20)

    model.save("./saved_model/unet5")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./figure/UNet - Model loss3.jpg')


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    restart = True
    while(restart):
        mode = input("Which network to be trained? (unet/segnet/segunet): ")
        if mode == 'unet' or mode == 'UNet':
            print("Training for U-Net:")
            mode = input("Training from exist model? (y/n): ")
            restart = False
            if mode == "y":
                train_from_model()
            elif mode == 'n':
                unet_train_new_v1()
            else:
                print("Invalid input.")
                restart = True
        elif mode == 'segnet' or mode == 'SegNet':
            print("Training for SegNet:")
            mode = input("Training from exist model? (y/n): ")
            restart = False
            if mode == "y":
                train_from_model()
            elif mode == 'n':
                segnet_train()
            else:
                print("Invalid input.")
                restart = True
        elif mode == 'segunet' or mode == 'SegUNet':
            print("Training for SegUNet:")
            mode = input("Training from exist model? (y/n): ")
            restart = False
            if mode == "y":
                train_from_model()
            elif mode == 'n':
                segunet_train()
            else:
                print("Invalid input.")
                restart = True
        else:
            restart = True
