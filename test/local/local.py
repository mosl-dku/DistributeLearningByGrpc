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

EPOCHS = 100
rmse_result = list()

# Data processing
data = pd.read_csv('data10.1.csv')
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

data_col = data.columns
x_train.columns = data_col

y_train = x_train.pop('LDL-C')
y_test = x_test.pop('LDL-C')

# Model load & Compile
model = makemodel.test_base(x_train)
model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])

# Model training
history2 = model.fit(x_train, 
                        y_train, 
                        batch_size=2048, 
                        epochs=EPOCHS,
			            verbose = 0
			)                       

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

results = model.evaluate(x_test, y_test, verbose = 0)

f = open("local_10.csv", "a")

for i in history2.history['loss']:
    f.write(str(i) + ',')
f.write(str(results[0]) + ',' + str(differ_sum) + ',' + str(round(differ_avg,3)) + '\n')
f.close()

