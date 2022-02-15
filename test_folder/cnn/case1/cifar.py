import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.compat.v1.Session(config=config)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

from conmodel import conmodel
from keras.models import load_model
from keras.models import Model


##데이터 받기 
cifar10 = datasets.cifar10 
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
 

##데이터 전처리
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
 
print("Train samples:", train_images.shape, train_labels.shape)
print("Test samples:", test_images.shape, test_labels.shape)


train_images = train_images.reshape((50000, 32, 32, 3))
test_images = test_images.reshape((10000, 32, 32, 3))
 
 
train_images = train_images/255.0
test_images = test_images/255.0
 
###############case 1##############

model = conmodel.model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_images, train_labels, epochs=10, verbose = 0)

test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('Test accuracy:', test_acc)
predictions = model.predict(test_images)

f = open("cnn_case1(loss).csv", "a")
for i in hist.history['loss'] :
    f.write(str(i) + ',')
f.write('\n')
f.close()

f = open("cnn_case1(acc).csv", "a")
for i in hist.history['accuracy'] :
    f.write(str(i * 100) + ',')
f.write('\n')
f.close()

f = open("cnn_case1(result).csv", "a")
f.write(str(test_acc * 100) + '\n')
f.close()
