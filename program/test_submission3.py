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

training_data = pd.read_csv('data/numerai_20180530/numerai_training_data.csv', header=0)
tournament_data = pd.read_csv('data/numerai_20180530/numerai_tournament_data.csv', header=0)
example_prediction = pd.read_csv('data/numerai_20180530/example_predictions.csv', header=0)

validation_data = tournament_data[tournament_data.data_type == 'validation']

X_train = training_data.drop(['id', 'era', 'data_type', 'target'], axis = 1).values
y_train = training_data.target.values

X_validation = validation_data.drop(['id', 'era', 'data_type', 'target'], axis = 1).values
y_validation = validation_data.target.values

X_train2 = np.concatenate([X_train, np.power(X_train, 2)], axis=1)
X_validation2 = np.concatenate([X_validation, np.power(X_validation, 2)], axis=1)


tic = time()
clf = lgb.LGBMClassifier(
        n_estimators = 100,
        num_leaves = 100,
        learning_rate = 0.01
        )
clf.fit(X_train2, y_train)
print(time()-tic)

clf.score(X_validation2, y_validation)
validation_pred_proba =  clf.predict_proba(X_validation2)
log_loss(y_validation, validation_pred_proba)



exported_pipeline = LogisticRegression(C=0.0001, dual=True, penalty="l2")

exported_pipeline.fit(X_train2, y_train)
pred = exported_pipeline.predict(X_validation2)

accuracy_score(y_validation, pred)
pred_proba = exported_pipeline.predict_proba(X_validation2)

log_loss(y_validation, pred_proba)

log_loss(y_validation, (validation_pred_proba + pred_proba)/2)


X_train3 = np.concatenate([X_train, np.power(X_train, 2), np.power(X_train, 3)], axis=1)
X_validation3 = np.concatenate([X_validation, np.power(X_validation, 2), np.power(X_validation, 3)], axis=1)

exported_pipeline.fit(X_train3, y_train)
pred = exported_pipeline.predict(X_validation3)

accuracy_score(y_validation, pred)
pred_proba = exported_pipeline.predict_proba(X_validation3)

log_loss(y_validation, pred_proba)


log_loss(y_validation, 0.7*validation_pred_proba + 0.3*pred_proba)
# 0.69260402730219628


tic = time()
clf = RandomForestClassifier(
        n_estimators = 50,
        min_samples_split = 8000,
        random_state = 7)
clf.fit(X_train, y_train)
print(time()-tic)

clf.score(X_validation, y_validation)
rf_pred_proba =  clf.predict_proba(X_validation)
log_loss(y_validation, rf_pred_proba)
# 0.69248324475642309
log_loss(y_validation, 0.7*rf_pred_proba + 0.3*validation_pred_proba)

X_submission = tournament_data.drop(['id', 'era', 'data_type', 'target'], axis = 1).values
submission_pred_proba =  clf.predict_proba(X_submission)
test_submission = pd.DataFrame({'id': tournament_data.id, 'probability': submission_pred_proba[:,1]})
test_submission.to_csv('submission/test_submission_20180430_4.csv', index=False)



for i in range(15):
    clf = RandomForestClassifier(
            n_estimators = 50,
            min_samples_split = 1000*(i+1),
            random_state = 1)
    clf.fit(X_train, y_train)
    
    rf_pred_proba =  clf.predict_proba(X_validation)
    print(i)
    print(log_loss(y_validation, rf_pred_proba))


clf = RandomForestClassifier(
        n_estimators = 50,
        min_samples_split = 6000,
        random_state = 1)
clf.fit(X_train, y_train)

rf_pred_proba =  clf.predict_proba(X_validation)
print(log_loss(y_validation, rf_pred_proba))

X_submission = tournament_data.drop(['id', 'era', 'data_type', 'target'], axis = 1).values
submission_pred_proba =  clf.predict_proba(X_submission)
test_submission = pd.DataFrame({'id': tournament_data.id, 'probability': submission_pred_proba[:,1]})
test_submission.to_csv('submission/test_submission_20180430_5.csv', index=False)



gridParams = {
    'n_estimators': [50,100,200,500],
    'num_leaves': [10,50,100],
    'learning_rate': [0.1, 0.01, 0.005]
    }

grid_loss = np.empty([4,3,3])

for i in range(4):
    for j in range(3):
        for k in range(3):
            clf = lgb.LGBMClassifier(
                n_estimators = gridParams['n_estimators'][i],
                num_leaves = gridParams['num_leaves'][j],
                learning_rate = gridParams['learning_rate'][k]
                )
            clf.fit(X_train, y_train)
            
            lgb_pred_proba =  clf.predict_proba(X_validation)
            print('n_estimators: ' + str(gridParams['n_estimators'][i]))
            print('num_leaves: ' + str(gridParams['num_leaves'][j]))
            print('learning_rate: ' + str(gridParams['learning_rate'][k]))
            print(log_loss(y_validation, lgb_pred_proba))
            grid_loss[i,j,k] = log_loss(y_validation, lgb_pred_proba)

#n_estimators: 200
#num_leaves: 10
#learning_rate: 0.1
#0.692434207372

clf = lgb.LGBMClassifier(
                n_estimators = 200,
                num_leaves = 10,
                learning_rate = 0.1
                )
clf.fit(X_train, y_train)
            
lgb_pred_proba =  clf.predict_proba(X_validation)
log_loss(y_validation, lgb_pred_proba)
log_loss(y_validation, rf_pred_proba)

log_loss(y_validation, 0.55*rf_pred_proba+0.45*lgb_pred_proba)


submission_pred_proba2 =  clf.predict_proba(X_submission)
test_submission = pd.DataFrame({'id': tournament_data.id, 'probability': 0.55*submission_pred_proba[:,1]+0.45*submission_pred_proba2[:,1]})
test_submission.to_csv('submission/test_submission_20180430_6.csv', index=False)


