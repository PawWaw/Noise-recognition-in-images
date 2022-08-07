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
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers
from keras import models
from keras import layers
import random
import numpy as np 
from sklearn.utils import class_weight
from array import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
np.random.seed(1000)

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

image_size = 224

# load an image from file
train_dir = "C:/Users/pawel/Documents/MAGISTERKA/Data/Set-I"
validation_dir = "C:/Users/pawel/Documents/MAGISTERKA/Data/Set-II"

model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(7))
model.add(Activation('softmax'))

model.summary()

train_datagen = ImageDataGenerator(
      )
validation_datagen = ImageDataGenerator(
#     rescale=1./255
    )

# Change the batchsize according to your system RAM
train_batchsize = 16
val_batchsize = 4
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
model.save('Train1Val2AlexNet.h5')

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