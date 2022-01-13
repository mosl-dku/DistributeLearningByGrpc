from __future__ import print_function
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.compat.v1.Session(config=config)


import logging
import grpc
import service_pb2
import service_pb2_grpc


import numpy as np
import pandas as pd
import time
from keras import optimizers
from keras import callbacks
from sklearn.model_selection import train_test_split
from keras.models import Model
from mkmodel import makemodel
from keras.utils import plot_model 
from keras.models import load_model

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def run1():
        with grpc.insecure_channel('172.25.244.2:50050', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 16)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def run2():
        with grpc.insecure_channel('172.25.244.2:50051', options=[('grpc.max_send_message_length', 1024 * 1024 * 200), ('grpc.max_receive_message_length', 1024 * 1024 * 200), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 16)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def testset():
        optimizer = optimizers.Adam(0.01)
        early_stop = callbacks.EarlyStopping(
    monitor='val_mean_squared_logarithmic_error', patience=10)

        EPOCHS = 200
        rmse_result = list()

        test = pd.read_csv('test2.csv')
        test = test.dropna(axis=1)

        test = test.drop(['DATE'], axis=1)
        test = test.rename(columns= {'L3008': 'TC', 'L3062': 'HDL-C', 'L3061' : 'TG', 'L3068' : 'LDL-C'})

        x_test = test
        y_testt = x_test.pop('LDL-C')

        model1 = makemodel.test_base(x_test)
        model1.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])

        mergedmodel1 = Model(inputs=model1.input, outputs=model1.get_layer('layer2').output)


        x_test_passedd = mergedmodel1.predict(x_test)

        return x_test_passedd, y_testt




if __name__ == '__main__':

        logging.basicConfig()
        optimizer = optimizers.Adam(0.01)
        early_stop = callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=10)
        EPOCHS = 100

        rmse_result = list()

        test = pd.read_csv('test2.csv')
        test = test.dropna(axis=1)
        test = test.drop(['DATE'], axis=1)
        test = test.rename(columns= {'L3008' : 'TC',
                             'L3062' : 'HDL-C',
                             'L3061' : 'TG',
                             'L3068' : 'LDL-C'})
        x_test = test
        y_test = x_test.pop('LDL-C')
        
        #model1 = makemodel.test_base(x_test)
        #model1.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
        #mergedmodel  = Model(inputs=model1.input, outputs=model1.get_layer('layer2').output)
        #x_test_passed = mergedmodel1.predict(x_test)
           
        fullmodel = load_model('full_model.h1')
        mergedmodel = Model(inputs=fullmodel.input, outputs=fullmodel.get_layer('layer6').output)

        x_test_passed = mergedmodel.predict(x_test)
        x_train1_passed, y_train1 = run1()
        x_train2_passed, y_train2 = run2()
#        x_test_passed, y_test = testset()
 
        merged_x_train = np.concatenate([x_train1_passed, x_train2_passed], axis=0)
        
        merged_y_train = np.concatenate([y_train1,y_train2],axis=0)
        merged_y_train = pd.Series(merged_y_train, name='LDL-C') 
        merged_y_train = merged_y_train.astype(int)

        print(x_train1_passed)
        print(type(x_train1_passed))
        print(x_train1_passed.shape)

        print(x_train2_passed)
        print(type(x_train2_passed))
        print(x_train2_passed.shape)

        print("merged")
        print(merged_x_train)
        print(type(merged_x_train))
        print(merged_x_train.shape)

 
        print(y_train1)
        print(type(y_train1))
        print(y_train1.shape)

        print(y_train2)
        print(type(y_train2))
        print(y_train2.shape)

        print(merged_y_train)
        print(type(merged_y_train))
        print(merged_y_train.shape)

        print("test")
        print(x_test_passed)
        print(type(x_test_passed))
        print(x_test_passed.shape)

        print(y_test)
        print(type(y_test))
        print(y_test.shape)


        test_L1out = makemodel.test_L3out(merged_x_train)
        test_L1out.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
    
        test_L1out.summary()

        history2 = test_L1out.fit(merged_x_train,
                          merged_y_train,
                          batch_size=2048,
                          epochs=EPOCHS
                          )
        print(history2)
        predict = test_L1out.predict(x_test_passed)
        pp=predict.reshape(-1)
        pp=np.round(pp)
        pp=pp.astype('int64')
     
        original = np.array(y_test.values)
        oo = original.reshape(-1)

        op = pd.DataFrame({'original':oo, 'predict':pp})

        mm = op['original'] - op['predict']
        mm = mm.to_numpy()
        mmabs = abs(mm)
        op['differ'] = mm
        op['abs'] =  mmabs

        differ_sum = op['abs'].sum()
        differ_avg = differ_sum / 81508

        print(op)
        print(differ_sum)
        print(round(differ_avg,3))

        plt.title("Distribute training_layer6")
        plt.ylim(0,300)
        plt.plot(history2.history['loss'], label='loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss(mse)')
        plt.legend(loc='best')
        plt.show()
        plt.savefig('./layer6_loss.png')

        plt.title("Distribute training_layer4")
        plt.ylim(0,0.5)
        plt.plot(history2.history['mean_squared_logarithmic_error'], label='mean_squared_logarithmic_error')
        plt.xlabel('Epoch')
        plt.ylabel('Local msle')
        plt.legend(loc='best')
        plt.show()
        plt.savefig('./layer6_mlse.png')
