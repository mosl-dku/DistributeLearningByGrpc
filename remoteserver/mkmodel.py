import tensorflow as tf

from keras.layers import *
from keras.models import *
from keras.utils import *

class makemodel:
    def __init__(self):
        self.result = 0

    def base_model(input_data):
        model = Sequential()
        model.add(Dense(32, activation='sigmoid', input_shape=[input_data.shape[1]], name='layer1'))
        model.add(Dropout(0.1, name='layer2'))
        # model.add(Dense(128, activation='sigmoid', name='layer3'))
        # model.add(Dropout(0.1, name='layer4'))
        model.add(Dense(64, activation='sigmoid', name='layer5'))
        model.add(Dropout(0.1, name='layer6'))
        # model.add(Dense(32, activation='sigmoid', name='layer7'))
        # model.add(Dropout(0.1, name='layer8'))
        model.add(Dense(16, activation='sigmoid', name='layer9'))
        model.add(Dropout(0.1, name='layer10'))
        model.add(Dense(2, activation='sigmoid', name='layer11'))
        model.add(Dropout(0.1, name='layer12'))
        model.add(Dense(1, name='final_layer'))
                
        return model
    
    def layer1_out(input_data):
        model = Sequential()
        model.add(Dense(128, activation='sigmoid', input_shape=[input_data.shape[1]], name='layer3'))
        model.add(Dropout(0.1, name='layer4'))
        model.add(Dense(64, activation='sigmoid', name='layer5'))
        model.add(Dropout(0.1, name='layer6'))
        model.add(Dense(32, activation='sigmoid', name='layer7'))
        model.add(Dropout(0.1, name='layer8'))
        model.add(Dense(16, activation='sigmoid', name='layer9'))
        model.add(Dropout(0.1, name='layer10'))        
        model.add(Dense(2, activation='sigmoid', name='layer11'))
        model.add(Dropout(0.1, name='layer12'))
        model.add(Dense(1, name='final_layer'))
                
        return model
    
    def layer2_out(input_data):
        model = Sequential()        
        model.add(Dense(64, activation='sigmoid', input_shape=[input_data.shape[1]], name='layer5'))
        model.add(Dropout(0.1, name='layer6'))
        # model.add(Dense(32, activation='relu', name='layer7'))
        # model.add(Dropout(0.1, name='layer8'))
        model.add(Dense(16, activation='sigmoid', name='layer9'))
        model.add(Dropout(0.1, name='layer10'))        
        model.add(Dense(2, activation='sigmoid', name='layer11'))
        model.add(Dropout(0.1, name='layer12'))
        model.add(Dense(1, name='final_layer'))
                
        return model
    
    def layer3_out(input_data):
        model = Sequential()                
        model.add(Dense(32, activation='relu', input_shape=[input_data.shape[1]], name='layer7'))
        model.add(Dropout(0.1, name='layer8'))
        model.add(Dense(16, activation='relu', name='layer9'))
        model.add(Dropout(0.1, name='layer10'))        
        model.add(Dense(2, activation='relu', name='layer11'))
        model.add(Dropout(0.1, name='layer12'))
        model.add(Dense(1, name='final_layer'))
                
        return model
    
    def layer4_out(input_data):
        model = Sequential()                
        model.add(Dense(16, activation='relu', input_shape=[input_data.shape[1]], name='layer9'))
        model.add(Dropout(0.1, name='layer10'))        
        model.add(Dense(2, activation='relu', name='layer11'))
        model.add(Dropout(0.1, name='layer12'))
        model.add(Dense(1, name='final_layer'))
                
        return model
    
    def normal_relu_model(input_data):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=[input_data.shape[1]]))
        model.add(Dropout(0,1))
        model.add(Dense(64, activation='relu', input_shape=[input_data.shape[1]]))
        model.add(Dropout(0,1))
        model.add(Dense(16, activation='relu', input_shape=[input_data.shape[1]]))
        model.add(Dropout(0,1))
        model.add(Dense(2, activation='relu', input_shape=[input_data.shape[1]]))
        model.add(Dropout(0,1))
        model.add(Dense(1))
        
        return model
    
    def test_base(input_data):
        model = Sequential()
        model.add(Dense(128, input_shape=[input_data.shape[1]], name='layer1'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.1, name='layer2'))
        model.add(Dense(64, name='layer3'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.1, name='layer4'))
        model.add(Dense(16, name='layer5'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.1, name='layer6'))
        model.add(Dense(2, name='layer7'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(1, name='final_layer'))
                
        return model
    
    def test_L1out(input_data):
        model = Sequential()        
        model.add(Dense(64, input_shape=[input_data.shape[1]], name='layer1'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.1, name='layer2'))
        model.add(Dense(16, name='layer3'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.1, name='layer4'))        
        model.add(Dense(2, name='layer5'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(1, name='final_layer'))
        
        return model
    
    def test_base_enc(input_data):
        """
        model = Sequential()
        model.add(Dense(128, input_shape=[input_data.shape[1]], name='layer1'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.1, name='layer2'))
        model.add(Dense(64, name='layer3'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.1, name='layer4'))
        model.add(Dense(16, name='layer5'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.1, name='layer6'))
        model.add(Dense(2, name='layer7'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(1, name='final_layer'))
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=[input_data.shape[1]], name='layer1'),
            tf.keras.layers.Dropout(0.1, name='layer2'),
            tf.keras.layers.Dense(64, activation=tf.nn.relu, name='layer3'),
            tf.keras.layers.Dropout(0.1, name='layer4'),
            tf.keras.layers.Dense(16, activation=tf.nn.relu, name='layer5'),
            tf.keras.layers.Dropout(0.1, name='layer6'),
            tf.keras.layers.Dense(2, activation=tf.nn.relu, name='layer7'),            
            tf.keras.layers.Dense(1, name='final_layer')
        ])
                
        return model

"""    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 8,
                               strides=2,
                               padding='same',
                               activation='relu',
                               batch_input_shape=input_shape),
        tf.keras.layers.AveragePooling2D(2, 1),
        tf.keras.layers.Conv2D(32, 4,
                               strides=2,
                               padding='valid',
                               activation='relu'),
        tf.keras.layers.AveragePooling2D(2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, name='logit')
  ])
"""    