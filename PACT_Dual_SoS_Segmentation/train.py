# Basic configuration
bs = 12
saved_model = './saved_model/unet_with_dense'

# import data
import data_io

trainset_loader = data_io.ImageDataGenerator(batch_size=bs)
validset_loader = data_io.ImageDataGenerator(batch_size=bs, mode='valid')

# Training network
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.optimizers as optim
import network

print("Training for a new UNet model:")

model = network.UNet()
plot_model(model)
model.compile(loss=BinaryCrossentropy(from_logits=False), optimizer=optim.Adam())

reduce_lr = ReduceLROnPlateau(
    verbose=1, factor=0.1, min_delta=0.1, monitor='val_loss', patience=10,
    mode='auto', min_lr=0.00001)
stop_condition = EarlyStopping(
    monitor='val_loss', patience=21, min_delta=0.1)

history = model.fit(trainset_loader, batch_size=bs, epochs=100,
                    validation_data=validset_loader, workers=16, max_queue_size=16,
                    callbacks=[reduce_lr, stop_condition])

# Save model and training log
model.save(saved_model)

from matplotlib import pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./figure/UNet - Model loss.jpg')