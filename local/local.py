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

data_col = data.columns
x_train.columns = data_col

y_train = x_train.pop('LDL-C')
y_test = x_test.pop('LDL-C')


# Model load & Compile
model = makemodel.test_base_2(x_train)
model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])


# Model training
history2 = model.fit(x_train, 
                        y_train, 
                        batch_size=2048, 
                        epochs=EPOCHS
			)                       
					
print(history2)

#model.save('1_model.h1')

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
print(data.columns)
#op.to_csv("./local_csv/local_df9.csv",mode='w')
# Make Graph

plt.title("Local training")
plt.ylim(0,300)
plt.plot(history2.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss(mse)')
plt.legend(loc='best')
plt.show()
plt.savefig('./local_loss.png')

plt.title("Local training")
plt.ylim(0,0.5)
plt.plot(history2.history['mean_squared_logarithmic_error'], label='mean_squared_logarithmic_error')
plt.xlabel('Epoch')
plt.ylabel('Local msle')
plt.legend(loc='best')
plt.show()
plt.savefig('./local_mlse.png')



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
plt.savefig('./local.png')
'''
