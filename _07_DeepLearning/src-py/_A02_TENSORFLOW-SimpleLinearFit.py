#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:15:57 2018

Using Tensorflow, we will create a neuron that performs a very simple linear fit to some
2-D Data

Steps are 
- Build a Graph
- Initiate the Session
- Feed data in and get output

- Then we can add in the cost function in order to train your network to optimize the params

@author: suvosmac
"""

# Required libraries
import numpy as np
import tensorflow as tf

np.random.seed(101)
tf.set_random_seed(101)

# Lets have two random numbers
rand_a = np.random.uniform(0,100,(5,5))
print(rand_a)

rand_b = np.random.uniform(0,100,(5,1))
print(rand_b)

# Now we will create placeholders for these random numbers
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Now we will add operations
add_op = a + b
mul_op = a * b

# We will now run operations on these placeholders using feed dictionaries which are 
# like providing data to placeholders
with tf.Session() as sess:
    add_result = sess.run(add_op,feed_dict={a:rand_a,b:rand_b})
    print(add_result)
    print("\n")
    mult_result = sess.run(mul_op,feed_dict={a:rand_a,b:rand_b})
    print(mult_result)

'''
Example Neural Network
'''

n_features = 10
n_dense_neurons = 3

# We will create a place holder - None because we dont know from now on how many training samples
# we will feed, but we know about the number of features which is the other dimension
x = tf.placeholder(tf.float32,shape = (None,n_features))

# For weight variable
W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))

# for bias variable
b = tf.Variable(tf.ones([n_dense_neurons]))

# Frame the equation
xW = tf.matmul(x,W)
z = tf.add(xW,b)

# We will pass this on to some sort of activation function
a = tf.sigmoid(z)

# run of all of these in a session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # result of each of these neurons
    layer_out = sess.run(a, feed_dict={x:np.random.random([1,n_features])})
print(layer_out)

'''
Simple Regression Example
'''
# We create a data as linearly spaced points between 1 to 10 and then add some noise to it
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
print(x_data)

y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
print(y_label)

# now we will plot
import matplotlib.pyplot as plt
plt.plot(x_data,y_label,'*')
plt.show()

# We can see some linear trend with noise, we will try to model this with a neural network
# we will solve y = mx + b

# initialize m with some random variable
m = tf.Variable(0.49)
b = tf.Variable(0.78)

# Cost function
error = 0
for x,y in zip(x_data,y_label):
    y_hat = m*x + b
    
    # This error is going to be large because m and b is totally random
    error += (y - y_hat)**2
    
# Now we need to minimize the error. To minimize the error, we need optimizer
# learning rate decides how fast we will descent on the optimizer.. too large learning rate
# we will overshoot the actual minimal, too small, it will take a long time to perform the minimize

# different learning rates are used for different problems
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    training_steps = 100
    for i in range(training_steps):
        sess.run(train)
    
    final_slope, final_intercept = sess.run([m,b])

# evaluate the results
x_test = np.linspace(0,10,10)
# y = mx + b
y_pred_plot = final_slope*x_test + final_intercept
plt.plot(x_test,y_pred_plot,'r')
plt.plot(x_data,y_label,'*')
plt.show()