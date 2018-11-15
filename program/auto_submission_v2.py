import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer

import os
os.chdir('C:/Users/n0269042/Documents/numerai')
data_path = 'numerai_20180919'

target_names = ['bernie', 'charles', 'elizabeth', 'jordan', 'ken']

training_data = pd.read_csv('data/' + data_path + '/numerai_training_data.csv', header=0)
tournament_data = pd.read_csv('data/' + data_path + '/numerai_tournament_data.csv', header=0)
example_prediction = pd.read_csv('data/' + data_path + '/example_predictions_target_bernie.csv', header=0)

validation_data = tournament_data[tournament_data.data_type == 'validation']

target_vars = [col for col in training_data.columns if col.find('target_') > -1]
feature_vars = [col for col in training_data.columns if col.find('feature') > -1]


for name in target_names:
    
    X_train = training_data.drop(['id', 'era', 'data_type'] + target_vars, axis = 1).values
    y_era_train = pd.Categorical(training_data.era)
    
    y_train = training_data['target_'+name].values
    
    X_validation = validation_data.drop(['id', 'era', 'data_type'] + target_vars, axis = 1).values
    y_validation = validation_data['target_'+name].values
    
    
    model_data = training_data    #.append(validation_data)
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
    
    
    enc = LabelBinarizer()
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
    pred_proba_lgb = final_model_lgb.predict_proba(X_era_tournament)
    
    
    final_model_rf = RandomForestClassifier(
            n_estimators = 50,
            min_samples_split = 6000,
            random_state = 1)
    final_model_rf.fit(X_era_model, y_model)
    pred_proba_rf =  final_model_rf.predict_proba(X_era_tournament)
    
    test_submission = pd.DataFrame({'id': tournament_data.id, 'probability_'+name: 0.5*pred_proba_lgb[:,1]+0.5*pred_proba_rf[:,1]})
    test_submission.to_csv('submission/' + name + '_submission_' + data_path + '.csv', index=False)
