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


 
cifar10 = datasets.cifar10 
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
 
print("Train samples:", train_images.shape, train_labels.shape)
print("Test samples:", test_images.shape, test_labels.shape)

train_images = train_images.reshape((50000, 32, 32, 3))
test_images = test_images.reshape((10000, 32, 32, 3))

train_images = train_images/255.0
test_images = test_images/255.0

# ###############remote##############

model = conmodel.model1_1(train_images)
model.load_weights('init_weight_soft.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs =5)
pred_model = Model(inputs=model.input, outputs = model.get_layer('max_pooling2d').output)
train_passed = pred_model.predict(train_images)
print("##############remote finish##############")

##########################main#####################################################

model_test = conmodel.model1_1(test_images)
model_test.load_weights('init_weight_soft.h5')
model_test.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_test.fit(test_images, test_labels, epochs =5)
test_passed_model = Model(inputs=model_test.input, outputs = model_test.get_layer('max_pooling2d_1').output)
test_passed = test_passed_model.predict(test_images)
print("##############test set finish##############")
print("##############               ##############")
print("##############               ##############")
print("############mainserver train start#########")

model3 = conmodel.model2_2()
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model3.fit(train_passed, train_labels, epochs=10, verbose = 0)

test_loss, test_acc = model3.evaluate(test_passed, test_labels)
 
# print('Test accuracy:', test_acc)
# predictions = model.predict(test_images)

f = open("cnn_case3_nodis(loss).csv", "a")
for i in hist.history['loss'] :
    f.write(str(i) + ',')
f.write('\n')
f.close()

f = open("cnn_case3_nodis(acc).csv", "a")
for i in hist.history['accuracy'] :
    f.write(str(i * 100) + ',')
f.write('\n')
f.close()

f = open("cnn_case3_nodis(result).csv", "a")
f.write(str(test_acc * 100) + '\n')
f.close()