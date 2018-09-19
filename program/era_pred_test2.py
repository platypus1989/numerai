import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from time import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from joblib import Parallel, delayed

import os
os.chdir('C:/Users/n0269042/Documents/numerai')

training_data = pd.read_csv('data/numerai_20180724/numerai_training_data.csv', header=0)
tournament_data = pd.read_csv('data/numerai_20180724/numerai_tournament_data.csv', header=0)
example_prediction = pd.read_csv('data/numerai_20180724/example_predictions_target_bernie.csv', header=0)

validation_data = tournament_data[tournament_data.data_type == 'validation']

target_vars = [col for col in training_data.columns if col.find('target_') > -1]
feature_vars = [col for col in training_data.columns if col.find('feature') > -1]

X_train = training_data.drop(['id', 'era', 'data_type'] + target_vars, axis = 1).values
y_era_train = pd.Categorical(training_data.era)
y_train = training_data.target_bernie.values

X_validation = validation_data.drop(['id', 'era', 'data_type'] + target_vars, axis = 1).values
y_validation = validation_data.target_bernie.values


rng = np.random.seed(1)
train_train_index = np.random.choice(X_train.shape[0], round(X_train.shape[0]*0.9), replace = False)
X_train_train = X_train[train_train_index, :]
y_era_train_train = y_era_train[train_train_index]
X_train_test = np.delete(X_train, train_train_index, 0)
y_era_train_test = np.delete(y_era_train, train_train_index, 0)


tic = time()
clf = lgb.LGBMClassifier(
        n_estimators = 100,
        num_leaves = 100,
        learning_rate = 0.01
        )
clf.fit(X_train_train, y_era_train_train)
print(time()-tic)
# 445.661048412323

clf.score(X_train_test, y_era_train_test)
# 0.011280201214400041


tic = time()
clf = lgb.LGBMClassifier(
        n_estimators = 100,
        num_leaves = 100,
        learning_rate = 0.01
        )
clf.fit(X_train, y_era_train)
print(time()-tic)
# 445.661048412323

y_era_validation_pred = clf.predict(X_validation)


enc = LabelBinarizer()
y_era_train_mat = enc.fit_transform(y_era_train)
y_era_validation_mat = enc.fit_transform(y_era_validation_pred)

X_era_train = np.hstack([X_train, y_era_train_mat])
X_era_validation = np.hstack([X_validation, y_era_validation_mat])

tic = time()
clf = lgb.LGBMClassifier(
        n_estimators = 1000,
        num_leaves = 100,
        learning_rate = 0.001
        )
clf.fit(X_era_train, y_train)
print(time()-tic)

clf.score(X_era_validation, y_validation)


model_data = training_data.append(validation_data)
X_model = model_data[feature_vars].values
y_era_model = pd.Categorical(model_data.era)

X_tournament = tournament_data[feature_vars].values

tic = time()
clf = lgb.LGBMClassifier(
        n_estimators = 100,
        num_leaves = 100,
        learning_rate = 0.01
        )
clf.fit(X_model, y_era_model)
print(time()-tic)

y_era_tournament_pred = clf.predict(X_tournament)
y_era_tournament_predict_proba = clf.predict_proba(X_tournament)



y_era_model_mat = enc.fit_transform(y_era_model)
y_era_tournament_mat = enc.fit_transform(y_era_tournament_pred)

X_era_model = np.hstack([X_model, y_era_model_mat])
X_era_tournament = np.hstack([X_tournament, y_era_tournament_mat])

y_model = model_data.target_bernie.values

tic = time()
final_model_lgb = lgb.LGBMClassifier(
        n_estimators = 1000,
        num_leaves = 100,
        learning_rate = 0.001
        )
final_model_lgb.fit(X_era_model, y_model)
print(time()-tic)

y_tournament_pred_proba = final_model_lgb.predict_proba(X_era_tournament)

test_submission = pd.DataFrame({'id': tournament_data.id, 'probability_bernie': y_tournament_pred_proba[:,1]})
test_submission.to_csv('submission/test_submission_20180724_1.csv', index=False)



final_model_rf = RandomForestClassifier(
        n_estimators = 50,
        min_samples_split = 6000,
        random_state = 1)
final_model_rf.fit(X_era_model, y_model)

y_tournament_pred_proba_rf =  final_model_rf.predict_proba(X_era_tournament)

test_submission = pd.DataFrame({'id': tournament_data.id, 'probability_bernie': 0.5*y_tournament_pred_proba[:,1]+0.5*y_tournament_pred_proba_rf[:,1]})
test_submission.to_csv('submission/test_submission_20180724_2.csv', index=False)




y_tournament_wtd_pred_proba = np.zeros([y_tournament_pred_proba.shape[0]])


era_template_mat = enc.fit_transform(y_era_model.categories)

tic = time()
for i in range(y_tournament_pred_proba.shape[0]):
    X_era_tournament_temp = np.hstack([np.repeat(X_tournament[i,:],132).reshape([50, 132]).transpose(), era_template_mat])
    temp_pred_proba = final_model_lgb.predict_proba(X_era_tournament_temp)

    y_tournament_wtd_pred_proba[i] = np.matmul(y_era_tournament_predict_proba[i,:], temp_pred_proba[:,1])
    
    if i==100: print("it will take approxmately {} seconds to finish".format((time()-tic)/100*y_tournament_pred_proba.shape[0]))

test_submission = pd.DataFrame({'id': tournament_data.id, 'probability_bernie': y_tournament_wtd_pred_proba})
test_submission.to_csv('submission/test_submission_20180727_1.csv', index=False)


def wtd_pred_proba(i):
    X_era_tournament_temp = np.hstack([np.repeat(X_tournament[i,:],132).reshape([50, 132]).transpose(), era_template_mat])
    temp_pred_proba = final_model_lgb.predict_proba(X_era_tournament_temp)
    return np.matmul(y_era_tournament_predict_proba[i,:], temp_pred_proba[:,1])

tic = time()
y_tournament_wtd_pred_proba2 = Parallel(n_jobs=4)(delayed(wtd_pred_proba)(i) for i in range(y_tournament_pred_proba.shape[0]))
print(time() - tic)


final_model_rf = RandomForestClassifier(
        n_estimators = 50,
        min_samples_split = 6000,
        random_state = 1)
final_model_rf.fit(X_era_model, y_model)

y_tournament_pred_proba_rf =  final_model_rf.predict_proba(X_era_tournament)

test_submission = pd.DataFrame({'id': tournament_data.id, 'probability_bernie': 0.5*y_tournament_pred_proba[:,1]+0.5*y_tournament_pred_proba_rf[:,1]})
test_submission.to_csv('submission/test_submission_20180724_2.csv', index=False)




