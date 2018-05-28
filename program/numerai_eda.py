import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model
import h2o
import os
os.chdir('C:/Users/n0269042/Documents/numerai')

training_data = pd.read_csv('data/numerai_20180423/numerai_training_data.csv', header=0)
prediction_data = pd.read_csv('data/numerai_20180423/numerai_tournament_data.csv', header=0)


train_data = training_data[training_data.era in ['era'+str(i+1) for i in range(80)]]
test_data = training_data[training_data.era in ['era'+str(i+41) for i in range(40)]]


h2o_train = h2o.H2OFrame(tr)


