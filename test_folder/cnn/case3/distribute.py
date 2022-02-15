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
label1 = train_labels[0:5000]

train2 = train_images[5000:10000]
train2 = train2.reshape((5000,32,32,3))
train2 = train2/255.0
label2 = train_labels[5000:10000]

train3 = train_images[10000:15000]
train3 = train3.reshape((5000,32,32,3))
train3 = train3/255.0
label3 = train_labels[10000:15000]

train4 = train_images[15000:20000]
train4 = train4.reshape((5000,32,32,3))
train4 = train4/255.0
label4 = train_labels[15000:20000]

train5 = train_images[20000:25000]
train5 = train5.reshape((5000,32,32,3))
train5 = train5/255.0
label5 = train_labels[20000:25000]

train6 = train_images[25000:30000]
train6 = train6.reshape((5000,32,32,3))
train6 = train6/255.0
label6 = train_labels[25000:30000]

train7 = train_images[30000:35000]
train7 = train7.reshape((5000,32,32,3))
train7 = train7/255.0
label7 = train_labels[30000:35000]

train8 = train_images[35000:40000]
train8 = train8.reshape((5000,32,32,3))
train8 = train8/255.0
label8 = train_labels[35000:40000]

train9 = train_images[40000:45000]
train9 = train9.reshape((5000,32,32,3))
train9 = train9/255.0
label9 = train_labels[40000:45000]

train10 = train_images[45000:50000]
train10 = train10.reshape((5000,32,32,3))
train10 = train10/255.0
label10 = train_labels[45000:50000]
test_images = test_images.reshape((10000, 32, 32, 3))

train_images = train_images/255.0
test_images = test_images/255.0

# ###############remote##############

model1 = conmodel.model1_1(train1)
model1.load_weights('init_weight_soft.h5')
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(train1, label1, epochs =5)
pred_model1 = Model(inputs=model1.input, outputs = model1.get_layer('max_pooling2d').output)
train1_passed = pred_model1.predict(train1)
print("##############train1 finish##############")

model2 = conmodel.model1_1(train2)
model2.load_weights('init_weight_soft.h5')
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(train2, label2, epochs =5)
pred_model2 = Model(inputs=model2.input, outputs = model2.get_layer('max_pooling2d_1').output)
train2_passed = pred_model2.predict(train2)
print("##############train2 finish##############")

model3 = conmodel.model1_1(train3)
model3.load_weights('init_weight_soft.h5')
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model3.fit(train3, label3, epochs =5)
pred_model3 = Model(inputs=model3.input, outputs = model3.get_layer('max_pooling2d_2').output)
train3_passed = pred_model3.predict(train3)
print("##############train3 finish##############")

model4 = conmodel.model1_1(train4)
model4.load_weights('init_weight_soft.h5')
model4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model4.fit(train4, label4, epochs =5)
pred_model4 = Model(inputs=model4.input, outputs = model4.get_layer('max_pooling2d_3').output)
train4_passed = pred_model4.predict(train4)
print("##############train4 finish##############")

model5 = conmodel.model1_1(train5)
model5.load_weights('init_weight_soft.h5')
model5.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model5.fit(train5, label5, epochs =5)
pred_model5 = Model(inputs=model5.input, outputs = model5.get_layer('max_pooling2d_4').output)
train5_passed = pred_model5.predict(train5)
print("##############train5 finish##############")

model6 = conmodel.model1_1(train6)
model6.load_weights('init_weight_soft.h5')
model6.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model6.fit(train6, label6, epochs =5)
pred_model6 = Model(inputs=model6.input, outputs = model6.get_layer('max_pooling2d_5').output)
train6_passed = pred_model6.predict(train6)
print("##############train6 finish##############")

model7 = conmodel.model1_1(train7)
model7.load_weights('init_weight_soft.h5')
model7.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model7.fit(train7, label7, epochs =5)
pred_model7 = Model(inputs=model7.input, outputs = model7.get_layer('max_pooling2d_6').output)
train7_passed = pred_model7.predict(train7)
print("##############train7 finish##############")

model8 = conmodel.model1_1(train8)
model8.load_weights('init_weight_soft.h5')
model8.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model8.fit(train8, label8, epochs =5)
pred_model8 = Model(inputs=model8.input, outputs = model8.get_layer('max_pooling2d_7').output)
train8_passed = pred_model8.predict(train8)
print("##############train8 finish##############")

model9 = conmodel.model1_1(train9)
model9.load_weights('init_weight_soft.h5')
model9.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model9.fit(train9, label9, epochs =5)
pred_model9 = Model(inputs=model9.input, outputs = model9.get_layer('max_pooling2d_8').output)
train9_passed = pred_model9.predict(train9)
print("##############train9 finish##############")

model10 = conmodel.model1_1(train10)
model10.load_weights('init_weight_soft.h5')
model10.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model10.fit(train10, label10, epochs =5)
pred_model10 = Model(inputs=model10.input, outputs = model10.get_layer('max_pooling2d_9').output)
train10_passed = pred_model10.predict(train10)
print("##############train10 finish##############")

##########################main#####################################################

model_test = conmodel.model1_1(test_images)
model_test.load_weights('init_weight_soft.h5')
model_test.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_test.fit(test_images, test_labels, epochs =5)
test_passed_model = Model(inputs=model_test.input, outputs = model_test.get_layer('max_pooling2d_10').output)
test_passed = test_passed_model.predict(test_images)
print("##############test set finish##############")
print("##############               ##############")
print("##############               ##############")
print("############mainserver train start#########")

merged_train = np.concatenate([train1_passed,train2_passed,train3_passed,train4_passed,train5_passed,train6_passed,train7_passed,train8_passed,train9_passed,train10_passed], axis=0)

model3 = conmodel.model2_2()
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model3.fit(merged_train, train_labels, epochs=10, verbose = 0)

test_loss, test_acc = model3.evaluate(test_passed, test_labels)
 
# print('Test accuracy:', test_acc)
# predictions = model.predict(test_images)

f = open("cnn_case3_dis(loss).csv", "a")
for i in hist.history['loss'] :
    f.write(str(i) + ',')
f.write('\n')
f.close()

f = open("cnn_case3_dis(acc).csv", "a")
for i in hist.history['accuracy'] :
    f.write(str(i * 100) + ',')
f.write('\n')
f.close()

f = open("cnn_case3_dis(result).csv", "a")
f.write(str(test_acc * 100) + '\n')
f.close()