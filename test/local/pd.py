import pandas as pd
import numpy as np


data = pd.read_csv('data.csv')

one = pd.DataFrame(np.array(data)[:round(int(len(data)*0.5))])
two = pd.DataFrame(np.array(data)[round(int(len(data)*0.5)):round(int(len(data)))])

data_col=data.columns

one.columns = data_col
two.columns = data_col
one.to_csv('one.csv',index=False)
two.to_csv('two.csv',index=False)

