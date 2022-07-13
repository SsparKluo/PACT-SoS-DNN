from matplotlib import pyplot as plt
import network
import tensorflow.keras.optimizers as optim
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import data_io
import os
from tensorflow.keras.models import load_model

# Basic configuration
bs = 12
model_name = 'cnn_dense_FT'
saved_model = './saved_model/{}'.format(model_name)
best_checkpoint = './saved_model/{}_best_2'.format(model_name)
figure_path = './figure/{} - Model loss.jpg/'.format(model_name)

if not os.path.exists(saved_model):
    os.mkdir(saved_model)
if not os.path.exists(best_checkpoint):
    os.mkdir(best_checkpoint)

# import data
trainset_loader = data_io.ImageDataGenerator(batch_size=bs)
validset_loader = data_io.ImageDataGenerator(batch_size=bs, mode='valid')

# Training network
print("Training for a new UNet model:")

decoder_model = load_model('./saved_model/unet_dense_2')
encoder_model = load_model('./saved_model/unet_forFT')
model = network.unet_with_dense(img_shape=(256,192,1), batchnorm=True)

for idx, layer in enumerate(model.layers):
    if 'up' in layer.name:
        break
    if 'conv2d' not in layer.name:
        continue
    layer.set_weights(encoder_model.layers[idx].get_weights())
    layer.trainable = False

reverse_layers = model.layers[::-1]

for idx, layer in enumerate(reverse_layers, start=1):
    if idx > 7:
        break
    if 'conv2d' not in layer.name:
        continue
    layer.set_weights(encoder_model.layers[-idx].get_weights())
    layer.trainable = False

model.summary()

model.compile(loss=BinaryCrossentropy(from_logits=False),
              optimizer=optim.Adam(learning_rate=0.0005))

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

history = model.fit(trainset_loader, batch_size=bs, epochs=200,
                    validation_data=validset_loader, workers=16, max_queue_size=16)

# Save model and training log
model.save(saved_model)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(figure_path)
