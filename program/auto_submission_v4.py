import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Subset

def logloss(true, pred_proba):
    return -(true*np.log(pred_proba) + (1-true)*np.log(1-pred_proba)).mean()

import os
os.chdir('C:/Users/n0269042/Documents/numerai')
data_path = 'numerai_20180925'

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
        
    y_era_model_pred_proba = clf.predict_proba(X_model)
    y_era_tournament_pred_proba = clf.predict_proba(X_tournament)
    
#    enc = LabelBinarizer()
#    y_era_model_mat = enc.fit_transform(y_era_model)
#    y_era_tournament_mat = enc.fit_transform(y_era_tournament_pred)
    
    X_era_model = np.hstack([X_model, y_era_model_pred_proba])
    X_era_tournament = np.hstack([X_tournament, y_era_tournament_pred_proba])
    
    y_model = model_data['target_'+name].values
    
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
    
    
        
    class Net(nn.Module):
    
        def __init__(self):
            super(Net, self).__init__()
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(X_era_model.shape[1], 1000)
            self.fc2 = nn.Linear(1000, 200)
            self.fc3 = nn.Linear(200, 1)
            self.fc4 = nn.Sigmoid()
    
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            x = self.fc4(x)
            return x
    
    fc = Net()
    criterion = nn.BCELoss()# Mean Squared Loss
    l_rate = 0.001
    optimiser = torch.optim.Adam(fc.parameters(), lr = l_rate) #Stochastic Gradient Descent
    
    epochs = 50
    batch_size = 2**9
    N = X_era_model.shape[0]
    X_train_tensor = torch.from_numpy(X_era_model.astype(np.float32))
    y_train_tensor = torch.from_numpy(y_model.astype(np.float32))
    X_era_validation = X_era_tournament[tournament_data.data_type == 'validation', :]    
    
    for epoch in range(epochs):

        epoch +=1
        #increase the number of epochs by 1 every time
        for k in range(0, N, batch_size):
            inputs = Variable(X_train_tensor.narrow(0, k, min(batch_size, N-k)))
            labels = Variable(y_train_tensor.narrow(0, k, min(batch_size, N-k)))
        
            #clear grads as discussed in prev post
            optimiser.zero_grad()
            #forward to get predicted values
            outputs = fc.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()# back props
            optimiser.step()# update the parameters
        pred = fc(X_train_tensor).data.numpy()[:,0]    
        print('epoch {}, train logloss {}'.format(epoch, logloss(y_model, pred)))
    
        pred = fc(torch.from_numpy(X_era_validation.astype(np.float32))).data.numpy()[:,0]    
        print('epoch {}, validation logloss {}'.format(epoch, logloss(y_validation, pred)))
        
    pred_proba_nn = fc(torch.from_numpy(X_era_tournament.astype(np.float32))).data.numpy()[:,0]
    
    test_submission = pd.DataFrame({'id': tournament_data.id, 'probability_'+name: 0.34*pred_proba_lgb[:,1]+0.33*pred_proba_rf[:,1]})
    test_submission.to_csv('submission/' + name + '_submission_' + data_path + '_v4.csv', index=False)
