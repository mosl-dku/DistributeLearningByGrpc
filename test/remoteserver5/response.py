from concurrent import futures
import logging

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.compat.v1.Session(config=config)


import grpc

import service_pb2
import service_pb2_grpc

import pandas as pd
import numpy as np
from keras import optimizers
from keras import callbacks
from sklearn.model_selection import train_test_split
from keras.models import Model
from mkmodel import makemodel
from keras.models import load_model

class layer2_out(service_pb2_grpc.layer2_outServicer):

    def request(self, request, context):
        optimizer = optimizers.Adam(0.01)
        early_stop = callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=10)

        EPOCHS = 50
        rmse_result = list()

        data = pd.read_csv('data10.5.csv')
        data =data.dropna(axis=1)

        data = data.drop(['DATE'], axis=1)
        data = data.rename(columns= {'L3008' : 'TC',
                                    'L3062' : 'HDL-C',
                                    'L3061' : 'TG',
                                    'L3068' : 'LDL-C'})

        x_train = data
        y_train = x_train.pop('LDL-C')
     
        model = makemodel.test_layer2(x_train)
        model.load_weights("weight.h5")
        model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
        model.fit(x_train,y_train,batch_size=2048,epochs=EPOCHS)
   
#        mergedmodel = Model(inputs=model.input, outputs=model.get_layer('layer2').output)
#        mergedmodel.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
       # mergedmodel.fit(x_train,y_train,batch_size=2048,epochs=EPOCHS)        

        #fullmodel = load_model('full_model.h1')
        #mergedmodel = Model(inputs=fullmodel.input, outputs=fullmodel.get_layer('layer2').output)
        x_train_passed = model.predict(x_train)


        bx = np.array(x_train_passed,dtype=np.float32).tobytes()
        by = np.array(y_train,dtype=np.float32).tobytes()
        return service_pb2.DL_response(x_train=bx,y_train=by)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options=[('grpc.max_send_message_length', 1024 * 1024 * 200),('grpc.max_receive_message_length', 1024 * 1024 * 200),],)
    service_pb2_grpc.add_layer2_outServicer_to_server(layer2_out(), server)
    server.add_insecure_port('172.25.244.2:50054')
    server.start()
    server.wait_for_termination()
    return


import sys
if __name__ == '__main__':
    logging.basicConfig()
    serve()
    sys.exit()

