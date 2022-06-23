import tensorflow as tf
import data_io
import network
import os
import loss_func
import matplotlib.pyplot as plt


def train_new(dataset, refset):
    print("Train from the very beginning:")
    model = network.UNet((256, 256, 1), depth=2, start_ch=2)
    model.summary()

    model.compile(
        loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam()
    )

    #history = model.fit_generator(data_io.train_data_generator(32), 
    #    steps_per_epoch=135, epochs=5, validation_steps=1,
    #    validation_data=data_io.test_data_generator(32))

    history = model.fit(dataset, refset, batch_size=32, epochs=3, validation_split=0.2)

    #model.save("saved_model2")
    print(history.history['loss'])
    
    fig = plt.figure()
    plt.plot(history.history['loss'],label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig('net'+'loss.png')
    
    


def train_exist(dataset, refset):
    print("Train from an exist model:")
    model = tf.keras.models.load_model("../BME6008 - Dissertation/saved_model")
    model.summary()

    history = model.fit_generator(data_io.train_data_generator(32), 
        steps_per_epoch=120, epochs=1, validation_steps=1,
        validation_data=data_io.test_data_generator(32))

    # history = model.fit(dataset, refset, batch_size = 32, epochs=30, validation_split=0.2)

    # model.save("saved_model1")

    fig = plt.figure()
    plt.plot(history.history['loss'],label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig('net'+'loss.png')


if __name__ == "__main__":
    mode = input("Use existed model (1 or 0): ")

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    dataset, refset = data_io.load_data_train()
    print("The shape of dataset:", dataset.shape)

    if mode == "1":
        train_exist(dataset, refset)
    else:
        train_new(dataset, refset)
