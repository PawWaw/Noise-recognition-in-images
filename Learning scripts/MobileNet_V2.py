# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras import optimizers
from keras import models
from keras import layers
import random
import numpy as np 
from sklearn.utils import class_weight
from array import *

# def add_noise(img):
#     '''Add random noise to an image'''
#     VARIABILITY = 50
#     deviation = VARIABILITY*random.random()
#     noise = np.random.normal(0, deviation, img.shape)
#     img += noise
#     np.clip(img, 0., 255.)
#     return img

# def balanceData(data_generator):
#     data_generator.filenames.sort()
#     data_generator.filepaths.sort()
#     data_generator.classes.sort()
#     classes_indexes, counts = np.unique(data_generator.classes, return_counts=True)
#     elements_count = dict(zip(classes_indexes, counts))
#     min_value = counts.min()
#     min_key = classes_indexes[np.where(counts == min_value)].min()
#     indexes = np.arange(0)
#     for x in classes_indexes:
#         count_x = elements_count.get(x)
#         if count_x!=min_value:
#             difference = count_x - min_value
#             start_index = sum(counts[0:x])
#             indexes = np.append(indexes,np.linspace(start = start_index, stop = start_index+count_x,num= difference).astype(int))
#         else:
#             pass #do nothing
#     for index in sorted(indexes[:-1], reverse=True):
#         data_generator.filepaths.pop(index)
#         data_generator.filenames.pop(index)
#         data_generator.classes = np.delete(data_generator.classes,index)

# load the model
image_size = 224
preTrainedModel = MobileNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(image_size,image_size,3), pooling=None, classes=1000, alpha=0.35)

# load an image from file
train_dir = "C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-I"
validation_dir = "C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-II"
# Freeze the layers except the last 3 layers
for layer in preTrainedModel.layers[:-3]:
    layer.trainable = False
# Check the trainable status of the individual layers
for layer in preTrainedModel.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()
#Add the convolutional base model
model.add(preTrainedModel)
# Add new layers
# model.add(Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='valid', activation="relu"))
model.add(Activation('relu'))
model.add(layers.Flatten())
# model.add(layers.Dense(2048, activation='relu'))   
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=7, activation='softmax'))

model.summary()

train_datagen = ImageDataGenerator(
      )
validation_datagen = ImageDataGenerator(
    # rescale=1./255
    )

# Change the batchsize according to your system RAM
train_batchsize = 32
val_batchsize = 8
epochs_number = 20

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical',
        shuffle=True)

# balanceData(train_generator)

# train_generator.n = train_generator.classes.size
# print("Changed number of images "+str(train_generator.samples)+" to "+str(train_generator.n))
# train_generator.samples = train_generator.n

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# balanceData(validation_generator)
# validation_generator.n = validation_generator.classes.size
# print("Changed number of images "+str(validation_generator.samples)+" to "+str(validation_generator.n))
# validation_generator.samples = validation_generator.n

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-6),
              metrics=['accuracy'])

# checkpoint
filepath="HB_MobileNetV2_1e-7_0.4-{epoch:02d}-{val_accuracy:.3f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
callbacks_list = [checkpoint]

# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size,
      epochs=epochs_number,
      callbacks=callbacks_list,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)

# Save the model
model.save('HB_MobileNetV2_1e-7_0.4.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = np.arange(1,epochs_number + 1,1)

plt.plot(epochs, acc, 'b-', markersize=2, label='Dok??adno???? treningowa')
plt.plot(epochs, val_acc, 'r-', markersize=2, label='Dok??adno???? walidacyjna')
plt.title('Dok??adno???? treningowa i walidacyjna')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b-', markersize=2, label='Strata treningowa')
plt.plot(epochs, val_loss, 'r-', markersize=2, label='Strata walidacyjna')
plt.title('Strata treningowa i walidacyjna')
plt.legend()

plt.show()