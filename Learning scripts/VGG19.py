import tensorflow as tf

from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from keras import backend as K
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers
from keras import models
from keras import layers
import random
import numpy as np 
from sklearn.utils import class_weight
from array import *

# load the model
image_size = 224
preTrainedModel = VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(image_size,image_size,3), pooling=None, classes=1000)

# load an image from file
train_dir = "C:/Users/pawel/Documents/MAGISTERKA/Data/Set-I"
validation_dir = "C:/Users/pawel/Documents/MAGISTERKA/Data/Set-II"

# Freeze the layers except the last 4 layers
for layer in preTrainedModel.layers[:-3]:
    layer.trainable = False
# Check the trainable status of the individual layers
for layer in preTrainedModel.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()
#Add the vgg convolutional base model
model.add(preTrainedModel)
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=7, activation='softmax'))

model.summary()

train_datagen = ImageDataGenerator(
      )
validation_datagen = ImageDataGenerator(
    # rescale=1./255
    )

# Change the batchsize according to your system RAM
train_batchsize = 8
val_batchsize = 8
epochs_number = 20

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical',
        shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-5),
              metrics=['acc'])

# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=epochs_number,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)

# Save the model
model.save('Train1Val3VGG19.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = np.arange(1,epochs_number + 1,1)

plt.plot(epochs, acc, 'bo', markersize=3, label='Training acc')
plt.plot(epochs, val_acc, 'ro', markersize=3, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', markersize=3, label='Training loss')
plt.plot(epochs, val_loss, 'ro', markersize=3, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()