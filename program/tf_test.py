# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:49:25 2018

@author: n0269042
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from time import time
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
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

X_test = tournament_data.drop(['id', 'era', 'data_type', 'target'], axis = 1).values


onehot_encoder = OneHotEncoder(sparse=False)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

  
train_dataset = X_train.astype(np.float32)
train_labels = onehot_encoder.fit_transform(y_train.reshape([y_train.shape[0], 1]))

validation_dataset = X_validation.astype(np.float32)
validation_labels = onehot_encoder.fit_transform(y_validation.reshape([y_validation.shape[0], 1]))

test_dataset = X_test.astype(np.float32)

batch_size = 128
num_features = 50
num_hidden = 128
num_labels = 2

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, num_features))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_validation_dataset = tf.constant(validation_dataset)
  tf_test_dataset = tf.constant(test_dataset)

 # tf_new_input = tf.placeholder(tf.float32, shape=(1, num_features))  

  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [num_features, num_labels], stddev=0.1), name = 'w1')
  
  layer1_biases = tf.Variable(tf.zeros(
      [num_labels]), name = 'b1')
 
  logits = tf.matmul(tf_train_dataset, layer1_weights) + layer1_biases
  
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.1, global_step, 5000, 0.5)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.sigmoid(logits)
  validation_prediction = tf.nn.sigmoid(tf.matmul(tf_validation_dataset, layer1_weights) + layer1_biases) 
  test_prediction = tf.nn.sigmoid(tf.matmul(tf_test_dataset, layer1_weights) + layer1_biases)
 # new_prediction = tf.nn.softmax(tf.matmul(new_hidden, layer4_weights) + layer4_biases)


num_steps = 10001
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  saver = tf.train.Saver()  
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      #print('Minibatch accuracy: %f' % log_loss(batch_labels.argmax(axis=1), predictions))
      print('Test accuracy: %f' % log_loss(validation_labels.argmax(axis=1), validation_prediction.eval()))  
      #print(hidden.eval(feed_dict = feed_dict))  # weird...
  pred_proba = test_prediction.eval()
  save_path = saver.save(session, "C:/Users/n0269042/Documents/model.ckpt")
  print("Model saved in path: %s" % save_path)

test_submission = pd.DataFrame({'id': tournament_data.id, 'probability': pred_proba[:,1]})
test_submission.to_csv('submission/test_submission_20180530_1.csv', index=False)

