import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from time import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tpot import TPOTClassifier

import os
os.chdir('C:/Users/n0269042/Documents/numerai')

training_data = pd.read_csv('data/numerai_20180430/numerai_training_data.csv', header=0)
tournament_data = pd.read_csv('data/numerai_20180430/numerai_tournament_data.csv', header=0)
example_prediction = pd.read_csv('data/numerai_20180430/example_predictions.csv', header=0)

validation_data = tournament_data[tournament_data.data_type == 'validation']

X_train = training_data.drop(['id', 'era', 'data_type', 'target'], axis = 1).values
y_train = training_data.target.values

X_validation = validation_data.drop(['id', 'era', 'data_type', 'target'], axis = 1).values
y_validation = validation_data.target.values

tpot = TPOTClassifier(generations=1, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_validation, y_validation))
tpot.export('program/tpot_test_pipeline.py')

from sklearn.linear_model import LogisticRegression

exported_pipeline = LogisticRegression(C=0.0001, dual=True, penalty="l2")

exported_pipeline.fit(X_train, y_train)
pred = exported_pipeline.predict(X_validation)

accuracy_score(y_validation, pred)
pred_proba = exported_pipeline.predict_proba(X_validation)

log_loss(y_validation, pred_proba)
# 0.69280788424671291

tic = time()
clf = lgb.LGBMClassifier(
        n_estimators = 100,
        num_leaves = 100,
        learning_rate = 0.01
        )
clf.fit(X_train, y_train)
print(time()-tic)

clf.score(X_validation, y_validation)
validation_pred_proba =  clf.predict_proba(X_validation)
log_loss(y_validation, validation_pred_proba)
# 0.69261629490574428

log_loss(y_validation, (validation_pred_proba + pred_proba)/2)
# 0.69267395668678899


