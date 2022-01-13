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

                x_train = xb_train.reshape(-1, 128)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def run2():
        with grpc.insecure_channel('172.25.244.2:50051', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 128)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def run3():
        with grpc.insecure_channel('172.25.244.2:50052', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 128)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def run4():
        with grpc.insecure_channel('172.25.244.2:50053', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 128)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def run5():
        with grpc.insecure_channel('172.25.244.2:50054', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 128)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def run6():
        with grpc.insecure_channel('172.25.244.2:50055', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 128)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def run7():
        with grpc.insecure_channel('172.25.244.2:50056', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 128)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def run8():
        with grpc.insecure_channel('172.25.244.2:50057', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 128)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def run9():
        with grpc.insecure_channel('172.25.244.2:50058', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 128)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def run10():
        with grpc.insecure_channel('172.25.244.2:50059', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 128)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train

def testset():
        optimizer = optimizers.Adam(0.01)
        early_stop = callbacks.EarlyStopping(
    monitor='val_mean_squared_logarithmic_error', patience=10)

        EPOCHS = 100
        rmse_result = list()

        test = pd.read_csv('test2.csv')
        test = test.dropna(axis=1)

        test = test[(test['L3008'] < 650)]
        test = test[(test['HEIGHT'] < 250)]

        test = test.drop(['DATE'], axis=1)
        test = test.drop(['sex_M'], axis=1)
        test = test.rename(columns= {'L3008': 'TC', 'L3062': 'HDL-C', 'L3061' : 'TG', 'L3068' : 'LDL-C'})

        x_test = test
        y_test = x_test.pop('LDL-C')

        model = makemodel.test_layer2(x_test)
        model.load_weights("weight2.h5")
        model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
        model.fit(x_test, y_test, batch_size=2048, epochs=EPOCHS, verbose = 0)

        x_test_passedd = model.predict(x_test)

        return x_test_passedd, y_test

if __name__ == '__main__':

        logging.basicConfig()
        optimizer = optimizers.Adam(0.01)
        early_stop = callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=10)
        EPOCHS = 100

        rmse_result = list()

        x_test_passed, y_test = testset()
        x_train1_passed, y_train1 = run1()
        x_train2_passed, y_train2 = run2()
        x_train3_passed, y_train3 = run3()
        x_train4_passed, y_train4 = run4()
        x_train5_passed, y_train5 = run5()
        # x_train6_passed, y_train6 = run6()
        # x_train7_passed, y_train7 = run7()
        # x_train8_passed, y_train8 = run8()
        # x_train9_passed, y_train9 = run9()
        # x_train10_passed, y_train10 = run10()
 
        merged_x_train = np.concatenate([x_train1_passed, x_train2_passed, x_train3_passed, x_train4_passed, x_train5_passed], axis=0)#, x_train6_passed, x_train7_passed, x_train8_passed, x_train9_passed, x_train10_passed], axis=0)
        
        merged_y_train = np.concatenate([y_train1, y_train2, y_train3, y_train4, y_train5], axis=0)#, y_train6, y_train7, y_train8, y_train9, y_train10],axis=0)
        merged_y_train = pd.Series(merged_y_train, name='LDL-C') 
        merged_y_train = merged_y_train.astype(int)

        test_L1out = makemodel.test_L1out(merged_x_train)
        test_L1out.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
    
        #test_L1out.summary()

        history2 = test_L1out.fit(merged_x_train,
                          merged_y_train,
                          batch_size=2048,
                          epochs=EPOCHS,
                          verbose = 0,
                          validation_data=(x_test_passed, y_test)
                          )
        
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

        # print(op)
        # print(differ_sum)
        # print(round(differ_avg,3))

        results = test_L1out.evaluate(x_test_passed, y_test, verbose = 0)
        f = open("case3_5(validation, result)(again).csv", "a")
        for i in history2.history['val_loss']:
                f.write(str(i) + ',')
        f.write(str(results[0]) + ',' + str(differ_sum) + ',' + str(round(differ_avg,3)) + '\n')
        f.close()

        f = open("case3_5(loss)(again).csv", "a")
        for i in history2.history['loss'] :
                f.write(str(i) + ',')
        f.write('\n')
        f.close()