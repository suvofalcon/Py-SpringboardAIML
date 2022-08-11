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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation='relu', input_dim=13, units=7, kernel_initializer='uniform'))
    classifier.add(Dense(activation='relu', units=7, kernel_initializer='uniform'))
    classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=50, epochs=50)

from datetime import datetime
start = datetime.now()
accuracies = cross_val_score(estimator=classifier, X=features_train, y=target_train, cv=5, verbose=1)
end = datetime.now()
print("Time Taken :", (end-start))

print("Accuracy of the model")
print(accuracies.mean())
print("Variance of the model")
print(accuracies.std())