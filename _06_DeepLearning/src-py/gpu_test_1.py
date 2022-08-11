from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

credit_data = pd.read_csv("D:\ML-Datasets\Jigsaw\credit_default_data.csv")

BILL_AMOUNTS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
credit_data['AVG_BILL_AMT'] = credit_data[BILL_AMOUNTS].apply(lambda x:np.mean(x), axis=1)

PAY_AMOUNTS = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
credit_data['AVG_PAY_AMT'] = credit_data[PAY_AMOUNTS].apply(lambda x:np.mean(x), axis=1)

feature_cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'AVG_BILL_AMT', 'AVG_PAY_AMT']
features = credit_data[feature_cols].values
target = credit_data['default payment next month'].values

from sklearn.model_selection import train_test_split

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3,
                                                                           random_state=101)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from datetime import datetime

tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
  classifier = Sequential()
  classifier.add(Dense(activation='relu', input_dim=13, units=7, kernel_initializer='uniform'))
  classifier.add(Dense(activation='relu', units=7, kernel_initializer='uniform'))
  classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
start = datetime.now()
classifier.fit(x=features_train, y=target_train, batch_size=50, epochs=50, verbose=1)
end = datetime.now()

print("Time Taken :", (end-start))

print("Accuracy of the model")
print(accuracies.mean())
print("Variance of the model")
print(accuracies.std())