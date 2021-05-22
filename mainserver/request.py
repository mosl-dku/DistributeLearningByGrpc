from __future__ import print_function
import logging
import grpc
import service_pb2
import service_pb2_grpc


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import optimizers
from keras import callbacks
from sklearn.model_selection import train_test_split
from keras.models import Model
from mkmodel import makemodel


def run1():
	with grpc.insecure_channel('172.25.244.53:50051', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
		stub = service_pb2_grpc.layer2_outStub(channel)
		response = stub.request(service_pb2.DL_request(state=1))
		xb_train = np.frombuffer(response.x_train, dtype=np.float32)
		yb_train = np.frombuffer(response.y_train, dtype=np.float32)

		x_train = xb_train.reshape(-1, 128)
		y_train = pd.Series(yb_train, name='LDL-C')
		return x_train, y_train

def run2():
	with grpc.insecure_channel('172.25.244.42:50051', options=[('grpc.max_send_message_length', 1024 * 1024 * 200), ('grpc.max_receive_message_length', 1024 * 1024 * 200), ],) as channel:
		stub = service_pb2_grpc.layer2_outStub(channel)
		response = stub.request(service_pb2.DL_request(state=1))
		xb_train = np.frombuffer(response.x_train, dtype=np.float32)
		yb_train = np.frombuffer(response.y_train, dtype=np.float32)

		x_train = xb_train.reshape(-1, 128)
		y_train = pd.Series(yb_train, name='LDL-C')
		return x_train, y_train
'''
def run3():
	with grpc.insecure_channel('172.25.244.53:50051', options=[('grpc.max_send_message_length', 1024 * 1024 * 200), ('grpc.max_receive_message_length', 1024 * 1024 * 200), ],) as channel:
		stub = service_pb2_grpc.layer2_outStub(channel)
		response = stub.request(service_pb2.DL_request(state=1))
		x_train = np.frombuffer(response.x_train, dtype=np.float32)
		y_train = np.frombuffer(response.y_train, dtype=np.float32)

		x_train = x_train.reshape(-1, 128)
		y_train = pd.Series(y_train, name='LDL-C')
		return x_train, y_train
'''
def testset():
	optimizer = optimizers.Adam(0.01)
	early_stop = callbacks.EarlyStopping(
    monitor='val_mean_squared_logarithmic_error', patience=10)

	EPOCHS = 200
	rmse_result = list()

	data = pd.read_csv('testset.csv')
	data = data.dropna(axis=1)

	data = data.drop(['DATE'], axis=1)
	data = data.rename(columns= {'L3008': 'TC', 'L3062': 'HDL-C', 'L3061' : 'TG', 'L3068' : 'LDL-C'})

	x_test = data
	y_test = x_test.pop('LDL-C')

	model = makemodel.test_base(x_test)
	model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])

	mergedmodel = Model(inputs=model.input, outputs=model.get_layer('layer2').output)
	mergedmodel.summary()

	x_test_passed = mergedmodel.predict(x_test)
	return x_test_passed, y_test

if __name__ == '__main__':
        logging.basicConfig()
        optimizer = optimizers.Adam(0.01)
        early_stop = callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=10)
        EPOCHS = 200
        rmse_result = list()

        x_train1_passed, y_train1 = run1()
        x_train2_passed, y_train2 = run2()
	#x_train3_passed, y_train3 = run3()
        x_test_passed, y_test = testset()
        #print(x_test_passed);
        merged_x_train = np.concatenate([x_train1_passed, x_train2_passed], axis=0)
        merged_y_train = pd.concat([y_train1, y_train2], axis=0)
        #merged_y_train = merged_y_train.reset_index()

        test_L1out = makemodel.test_L1out(merged_x_train)
        test_L1out.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])

        test_L1out.summary()

        predictions = test_L1out.predict(x_test_passed)


        history2 = test_L1out.fit(merged_x_train,
                          merged_y_train,
                          batch_size=2048,
                          epochs=EPOCHS,
                          validation_data=(x_test_passed, y_test))
        print(history2)
