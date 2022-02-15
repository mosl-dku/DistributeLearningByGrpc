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
train1 = train_images[0:5000]
train1 = train1.reshape((5000,32,32,3))
train1 = train1/255.0
train2 = train_images[5000:10000]
train2 = train2.reshape((5000,32,32,3))
train2 = train2/255.0
train3 = train_images[10000:15000]
train3 = train3.reshape((5000,32,32,3))
train3 = train3/255.0
train4 = train_images[15000:20000]
train4 = train4.reshape((5000,32,32,3))
train4 = train4/255.0
train5 = train_images[20000:25000]
train5 = train5.reshape((5000,32,32,3))
train5 = train5/255.0
train6 = train_images[25000:30000]
train6 = train6.reshape((5000,32,32,3))
train6 = train6/255.0
train7 = train_images[30000:35000]
train7 = train7.reshape((5000,32,32,3))
train7 = train7/255.0
train8 = train_images[35000:40000]
train8 = train8.reshape((5000,32,32,3))
train8 = train8/255.0
train9 = train_images[40000:45000]
train9 = train9.reshape((5000,32,32,3))
train9 = train9/255.0
train10 = train_images[45000:50000]
train10 = train10.reshape((5000,32,32,3))
train10 = train10/255.0
test_images = test_images.reshape((10000, 32, 32, 3))

train_images = train_images/255.0
test_images = test_images/255.0
 

###############case 2##############

model = conmodel.model1(train_images)
model.load_weights('init_weight_2.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train1_passed = model.predict(train1)
train2_passed = model.predict(train2)
train3_passed = model.predict(train3)
train4_passed = model.predict(train4)
train5_passed = model.predict(train5)
train6_passed = model.predict(train6)
train7_passed = model.predict(train7)
train8_passed = model.predict(train8)
train9_passed = model.predict(train9)
train10_passed = model.predict(train10)

merged_train = np.concatenate([train1_passed,train2_passed,train3_passed,train4_passed,train5_passed,train6_passed,train7_passed,train8_passed,train9_passed,train10_passed], axis=0)

test_passed =  model.predict(test_images)

model2 = conmodel.model2(merged_train)
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model2.fit(merged_train, train_labels, epochs=10, verbose = 0)

test_loss, test_acc = model2.evaluate(test_passed, test_labels)
 
# print('Test accuracy:', test_acc)
# predictions = model.predict(test_images)

f = open("cnn_case2_dis(loss).csv", "a")
for i in hist.history['loss'] :
    f.write(str(i) + ',')
f.write('\n')
f.close()

f = open("cnn_case2_dis(acc).csv", "a")
for i in hist.history['accuracy'] :
    f.write(str(i * 100) + ',')
f.write('\n')
f.close()

f = open("cnn_case2_dis(result).csv", "a")
f.write(str(test_acc * 100) + '\n')
f.close()