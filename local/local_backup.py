# parameter set
import pandas as pd
import numpy as np
from keras import optimizers
from keras import callbacks
from sklearn.model_selection import train_test_split
from keras.models import Model
from mkmodel import makemodel
from keras.utils import plot_model

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


optimizer = optimizers.Adam(0.01)
early_stop = callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=10)

EPOCHS = 30
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

model = makemodel.test_base(x_train)
model1 = makemodel.test_base(x_test)
model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
model1.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])


history2 = model.fit(x_train, 
                        y_train, 
                        batch_size=2048, 
                        epochs=EPOCHS,                         
                        validation_data=(x_test, y_test))

print(history2)



# Predict & compare
predict = model.predict(x_test)
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

'''
# callbacks=[early_stop], 

print(history2)
plt.title("Layer2 Loss-Val Loss half dataset")
plt.ylim(0,500)
plt.plot(history2.history['loss'], label='loss')
plt.plot(history2.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')


plt.title("Layer2 mean_squared_logarithmic_error half dataset")
plt.ylim(0,0.1)
plt.plot(history2.history['mean_squared_logarithmic_error'], label='mean_squared_logarithmic_error')
plt.plot(history2.history['val_mean_squared_logarithmic_error'], label = 'val_mean_squared_logarithmic_error')
plt.xlabel('Epoch')
plt.ylabel('mean_squared_logarithmic_error')
plt.legend(loc='best')
plt.show()
plt.savefig('./l.png')

predictions = test_L1out.predict(x_test_passed)
print(predictions)

# mse, _ = test_L1out.evaluate(x_test_passed, y_test)
# rmse = np.sqrt(mse)
# print('Root Mean Square Error on test set: {}'.format(round(rmse, 3)))


# rmse_result.append(rmse)


test_base = makemodel.test_base(x_train)
test_base.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])
test_base.summary()

#merged_train = np.concatenate([x_train1, x_train2, x_train3], axis=0)

mergedmodel = Model(inputs=model.)



history1 = test_base.fit(merged_train, 
                        y_train, 
                        batch_size=2048, 
                        epochs=EPOCHS,                         
                        validation_data=(x_test, y_test))
# callbacks=[early_stop], 

mse, _ = test_base.evaluate(x_test, y_test)
rmse = np.sqrt(mse)
print('Root Mean Square Error on test set: {}'.format(round(rmse, 3)))
# rmse_result.append(rmse)
'''
