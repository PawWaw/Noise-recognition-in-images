import tensorflow as tf

from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import keras
from numpy import loadtxt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from numpy import argmax
from sklearn.metrics import confusion_matrix,classification_report
 
# load model
model = load_model('C:/Users/pawel/Documents/MAGISTERKA/Nets/HighBlur/MobileNet_V2/1/HB_MobileNetV2_1e-5_0.5-15-0.962.h5')
# summarize model.
model.summary()
# load dataset
test_dir = "C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-III"
test_datagen = ImageDataGenerator(
      )
    
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        shuffle=False)
# evaluate the model
score = model.evaluate_generator(test_generator, verbose=1)
# predicted_classes = argmax(score, axis=1)
# true_classes = test_generator.classes
# class_labels = list(test_generator.class_indices.keys()) 
# print(class_labels)

# print(confusion_matrix(true_classes, predicted_classes))
# report = classification_report(true_classes, predicted_classes, target_names=class_labels)
# print(report)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))