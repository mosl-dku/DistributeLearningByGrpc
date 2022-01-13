# parameter set
import pandas as pd
import numpy as np
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

optimizer = optimizers.Adam(0.01)
early_stop = callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=10)

EPOCHS = 5
rmse_result = list()

data = pd.read_csv('data.csv')
test = pd.read_csv('test2.csv')
data =data.dropna(axis=1)
test = test.dropna(axis=1)

data = data.drop(['DATE'], axis=1)
data = data.rename(columns= {'L3008' : 'TC',
                             'L3062' : 'HDL-C',
                             'L3061' : 'TG',
                             'L3068' : 'LDL-C'})
test = test.drop(['DATE'], axis=1)
test = test.rename(columns= {'L3008' : 'TC',
                             'L3062' : 'HDL-C',
                             'L3061' : 'TG',
                             'L3068' : 'LDL-C'})

x_train = data
x_test = test

y_train = x_train.pop('LDL-C')
y_test = x_test.pop('LDL-C')

x_train1 = pd.DataFrame(np.array(x_train)[:round(int(len(x_train)*0.5))])
x_train2 = pd.DataFrame(np.array(x_train)[round(int(len(x_train)*0.5)):round(int(len(x_train)))])

data_col = data.columns

x_train1.columns = data_col
x_train2.columns = data_col


model = makemodel.test_base(x_train1)
#model1 = makemodel.test_base(x_train2)
#model2 = makemodel.test_base(x_test)
#model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
#model1.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
#model2.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
#model1.fit(x_train,y_train,batch_size=2048,epochs=EPOCHS)


mergedmodel = Model(inputs=model.input, outputs=model.get_layer('layer2').output)
#mergedmodel1 = Model(inputs=model1.input, outputs=model1.get_layer('layer2').output)
#mergedmodel2 = Model(inputs=model2.input, outputs=model2.get_layer('layer2').output)
mergedmodel.summary()

#mergedmodel.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
#mergedmodel.fit(x_train, y_train, batch_size=2048,epochs=100)
#mergedmodel.save('merge.h1')
#fullmodel = load_model('full_model.h1')
#mergedmodel = Model(inputs=fullmodel.input, outputs=fullmodel.get_layer('layer2').output)
#mergedmodel.summary()


'''
x_train1_passed = mergedmodel.predict(x_train1)
x_train2_passed = mergedmodel.predict(x_train2)
x_test_passed = mergedmodel.predict(x_test)


merged_train = np.concatenate([x_train1_passed, x_train2_passed], axis=0)

print(x_train1_passed)
print(type(x_train1_passed))
print(x_train1_passed.shape)

print(x_train2_passed)
print(type(x_train2_passed))
print(x_train2_passed.shape)

print("merged")
print(merged_train)
print(type(merged_train))
print(merged_train.shape)

print(y_train)
print(type(y_train))
print(y_train.shape)

print("test")
print(x_test_passed)
print(type(x_test_passed))
print(x_test_passed.shape)

print(y_test)
print(type(y_test))
print(y_test.shape)

test_L1out = makemodel.test_L1out(merged_train)
test_L1out.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])

test_L1out.summary()

history2 = test_L1out.fit(merged_train, 
                        y_train, 
                        batch_size=2048, 
                        epochs=EPOCHS
			)

print(history2)

predict = test_L1out.predict(x_test_passed)
pp=predict.reshape(-1)
pp=np.round(pp)
pp=pp.astype('int64')
print(pp)
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

op.to_csv("./local_dis_csv/local_distribute_layer6_10.csv",mode='w')

# Make Graph

go = op['original'].loc[0:50]
gp = op['predict'].loc[0:50]

plt.figure(figsize=(20,5))
plt.title('local')

plt.plot(go.index, go.values, '_', markersize=6, color='blue', label='orig')
plt.plot(gp.index, gp.values, '_', markersize=6, color='salmon', label='pred')
plt.ylabel('LDL-C')
plt.xlabel('index')
plt.legend(loc='upper left')
plt.show()
plt.savefig('./local_distribute_l6.png')
'''
