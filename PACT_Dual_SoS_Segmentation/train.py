from matplotlib import pyplot as plt
import network
import tensorflow.keras.optimizers as optim
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import data_io
import os

# Basic configuration
bs = 12
saved_model = './saved_model/cnn_dense'
best_checkpoint = './saved_model/cnn_dense_best'
figure_path = './figure/'

if not os.path.exists(saved_model):
    os.mkdir(saved_model)
if not os.path.exists(best_checkpoint):
    os.mkdir(best_checkpoint)
if not os.path.exists(figure_path):
    os.mkdir(figure_path)

# import data

trainset_loader = data_io.ImageDataGenerator2(batch_size=bs)
validset_loader = data_io.ImageDataGenerator2(batch_size=bs, mode='valid')

# Training network

print("Training for a new UNet model:")

model = network.cnn_dense(img_shape=(256, 192, 1))
plot_model(model, show_shapes=True)
model.summary()
model.compile(loss=MeanSquaredError(),
              optimizer=optim.Adam(learning_rate=0.001))

reduce_lr = ReduceLROnPlateau(
    verbose=1, factor=0.1, min_delta=0.001, monitor='val_loss', patience=10,
    mode='auto', min_lr=0.00001)
stop_condition = EarlyStopping(
    monitor='val_loss', patience=21, min_delta=0.001)
checkpoint = ModelCheckpoint(filepath=best_checkpoint,
                             monitor='val_loss',
                             save_best_only='True',
                             mode='auto',
                             period=1)

history = model.fit(trainset_loader, batch_size=bs, epochs=400,
                    validation_data=validset_loader, workers=16, max_queue_size=16,
                    callbacks=[reduce_lr, stop_condition])

# Save model and training log
model.save(saved_model)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(figure_path + '/cnn_dense - Model loss.jpg')
