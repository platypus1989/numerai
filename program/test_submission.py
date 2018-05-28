import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from time import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

import os
os.chdir('C:/Users/n0269042/Documents/Competition/numerai')

training_data = pd.read_csv('data/numerai_20180430/numerai_training_data.csv', header=0)
tournament_data = pd.read_csv('data/numerai_20180430/numerai_tournament_data.csv', header=0)
example_prediction = pd.read_csv('data/numerai_20180430/example_predictions.csv', header=0)

validation_data = tournament_data[tournament_data.data_type == 'validation']

X_train = training_data.drop(['id', 'era', 'data_type', 'target'], axis = 1).values
y_train = training_data.target.values

X_validation = validation_data.drop(['id', 'era', 'data_type', 'target'], axis = 1).values
y_validation = validation_data.target.values


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

X_submission = tournament_data.drop(['id', 'era', 'data_type', 'target'], axis = 1).values
submission_pred_proba =  clf.predict_proba(X_submission)
test_submission = pd.DataFrame({'id': tournament_data.id, 'probability': submission_pred_proba[:,1]})
test_submission.to_csv('submission/test_submission_20180430_3.csv', index=False)


tic = time()
clf = lgb.LGBMClassifier(
        n_estimators = 100,
        num_leaves = 100,
        learning_rate = 0.01,
        min_child_samples = 100
        )
clf.fit(X_train, y_train)
print(time()-tic)

clf.score(X_validation, y_validation)
validation_pred_proba =  clf.predict_proba(X_validation)
log_loss(y_validation, validation_pred_proba)


tic = time()
clf = RandomForestClassifier(
        n_estimators = 50,
        min_samples_split = 10000)
clf.fit(X_train, y_train)
print(time()-tic)

clf.score(X_validation, y_validation)
validation_pred_proba =  clf.predict_proba(X_validation)
log_loss(y_validation, validation_pred_proba)

tic = time()
clf = LogisticRegression()
clf.fit(X_train, y_train)
print(time()-tic)

clf.score(X_validation, y_validation)
validation_pred_proba =  clf.predict_proba(X_validation)
log_loss(y_validation, validation_pred_proba)



tic = time()
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
print(time()-tic)
clf.score(X_validation, y_validation)


svm_parameters = {'gamma':[i+1 for i in range(5)], 'C':[0.2, 1, 5, 10]}
svc = SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)

sorted(clf.cv_results_.keys())



tic = time()
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
print(time()-tic)

tic = time()
print(clf.score(X_validation, y_validation))
print(time()-tic)



tic = time()
clf = RadiusNeighborsClassifier(radius = 1,
                                algorithm = 'ball_tree')
clf.fit(X_train, y_train)
print(time()-tic)

tic = time()
print(clf.score(X_validation[:10000,:], y_validation[:10000]))
print(time()-tic)

