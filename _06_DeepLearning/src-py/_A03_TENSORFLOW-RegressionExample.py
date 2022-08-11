# -*- coding: utf-8 -*-
"""
We will demonstrate tensor flow for a regression example

Here we will build the dataset

@author: suvosmac
"""

# import the needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Lets build the data
x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.randn(len(x_data))

# We will build the model on 
# y = mx + b
# we will set b = 5

y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])
# Check the initial rows
x_df.head()
y_df.head()

my_data = pd.concat([x_df,y_df],axis=1)
my_data.head()

# To plot this large data frame, we study a sample of 250
my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
plt.show() # shows a clear linear trend

# we will feed the training data in batches for neural network training
batch_size = 10 #(10 points at a time)

# initialize the slope and intercept to some random numbers
m = tf.Variable(0.5)
b = tf.Variable(1.0)

xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_model = m*xph + b

error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# initialize the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    # we are feeding 1000 batches of data, where each batch size is 10
    batches = 1000

    for i in range(batches):
        rand_ind = np.random.randint(len(x_data),size=batch_size)

        # The batches are also created random based on the rand_ind
        # The rand_ind choses the index for each of the batches
        feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}
        
        sess.run(train,feed_dict=feed)

    model_m, model_b = sess.run([m,b])

print("Slope is :",model_m)
print("Intercept is :",model_b)

# so the predicted values are
y_hat = x_data*model_m + model_b

# Now we will plot these predictions
my_data.sample(250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(x_data,y_hat,'r')
plt.show()

'''
How to use TF_ESTIMATOR
'''

# define the feature columns
feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
# Now we will define the estimator
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# do the train, test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y_true,
        test_size=0.3, random_state=101)

# Check the shape of the splits
print(x_train.shape)
print(x_test.shape)

# define the input function for the neural network
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=10,
                num_epochs=None,shuffle=True)

train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=10,
                num_epochs=1000,shuffle=False)

test_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_test},y_test,batch_size=10,
                num_epochs=None,shuffle=True)

estimator.train(input_fn=input_func,steps=1000)

# Now prediction and evaluation
train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)
eval_metrics = estimator.evaluate(input_fn=test_input_func,steps=1000)

print('Training Data Metrics')
print(train_metrics)

print('Evaluation Data Metrics')
print(eval_metrics)

'''
This is a good way to see, whether the model is overfitting on the training data
One of the way to check this, if the model is showing a low loss on training data and 
high loss on the eval data... 
Ideally loss of training and eval should be close to each other
'''

# for prediction - we will test it against a brand new data
brand_new_data = np.linspace(0,10,10)
input_fn_pred = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},shuffle=False)

# to see the predictions we need to cast it to a list, as it returns a generator object
# generator object - essentially a dictionary
list(estimator.predict(input_fn=input_fn_pred))

predictions = []
for pred in estimator.predict(input_fn=input_fn_pred):
    predictions.append(pred['predictions'])

# Check the predictions
print("The predictions are :")
print(predictions)

my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(brand_new_data,predictions,'r')
plt.show()
