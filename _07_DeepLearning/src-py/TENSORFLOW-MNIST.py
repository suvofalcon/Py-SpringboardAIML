#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:53:43 2018

MNIST with Multi Layer perceptron Model. we would use the MNIST dataset
using tensor flow

@author: suvosmac
"""
# import libraries
import tensorflow as tf

# Load the data (this is built into tensorflow)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data',one_hot=True)

#mnist is a tensorflow dataset object... we can grab attributes from this object as
type(mnist.train.images)
mnist.train.images.shape # Its an array of 55000 rows and 784 columns... 

# There are 55000 images, each with 28px X 28Px (784 pixels)

# to see one image in random
mnist.train.images[2].reshape(28,28)
# The values represents the amount of darkness in that pixel (ranging between 0 and 1)
# To visualize this using matplotlib

import matplotlib.pyplot as plt
plt.imshow(mnist.train.images[2].reshape(28,28),cmap='Greys')

'''
Now we will build a multi layer perceptron model to build a classification model
we will first define few key params to be used in the model

learning rate - how fast the cost function is adjusted. This can be adjusted for quickly the optimisation function is applied
lower the learning rate, higher the possibility for accurate training results at the cost of more time to train.
However after a certain extent the benefits are not seen, like diminishing returns

training_epochs - How many training cycles we go through
batch_size = size of the batches of training data

n_classes - Number of classes for classification, in this case it is 10 (0-9)
n_samples - Number of samples we have from the data
n_input - Number of input parameters
n_hidden_1 - How many neurons we want in the hidden layer 1
n_hidden_2 - How many neurons we want in the hiddent layer 2
'''
learning_rate = 0.001
training_epochs = 30
batch_size = 100
n_classes = 10
n_samples = mnist.train.num_examples
n_input = mnist.train.images.shape[1]
n_hidden_1 = 256
n_hidden_2 = 256

'''
Weights and Bias
In order for our tensorflow model to work we need to create two dictionaries containing our weight and bias objects 
for the model. We can use the tf.variable object type. This is different from a constant because TensorFlow's 
Graph Object becomes aware of the states of all the variables. A Variable is a modifiable tensor that lives in 
TensorFlow's graph of interacting operations. It can be used and even modified by the computation. 
We will generally have the model parameters be Variables. From the documentation string:
A variable maintains state in the graph across calls to `run()`. You add a variable to the graph by 
constructing an instance of the class `Variable`.

The `Variable()` constructor requires an initial value for the variable, which can be a `Tensor` of any type and shape. 
The initial value defines the type and shape of the variable. After construction, the type and shape of the 
variable are fixed. The value can be changed using one of the assign methods.
We'll use tf's built-in random_normal method to create the random values for our weights and biases 
(you could also just pass ones as the initial biases).
'''

# We are asking to generate random numbers in a matrix of n_input rows and n_hidden_1 columns and similarly for others

weights = {'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
           'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
           'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
           }

bias = {'b1':tf.Variable(tf.random_normal([n_hidden_1])),
        'b2':tf.Variable(tf.random_normal([n_hidden_2])),
        'out':tf.Variable(tf.random_normal([n_classes]))
        }

x = tf.placeholder('float',shape=[None,n_input])
y = tf.placeholder('float',shape=[None,n_classes])

'''
First we receive the input data array and then to send it to the first hidden layer. Then the data will begin to have 
a weight attached to it between layers (remember this is initially a random value) and then sent to a node to undergo 
an activation function (along with a Bias as mentioned in the lecture). Then it will continue on to the next hidden layer, 
and so on until the final output layer. In our case, we will just use two hidden layers, the more you use the longer the 
model will take to run (but it has more of an opportunity to possibly be more accurate on the training data).

Once the transformed "data" has reached the output layer we need to evaluate it. Here we will use a loss function 
(also called a cost function) to evaluate how far off we are from the desired result. In this case, how many of the 
classes we got correct.

Then we will apply an optimization function to minimize the cost (lower the error). This is done by adjusting weight 
values accordingly across the network. In out example, we will use the Adam Optimizer, which keep in mind, relative to 
other mathematical concepts, is an extremely recent development.
We can adjust how quickly to apply this optimization by changing our earlier learning rate parameter. 
The lower the rate the higher the possibility for accurate training results, but that comes at the cost of 
having to wait (physical time wise) for the results. Of course, after a certain point there is no benefit to lower 
the learning rate.

Now we will create our model, we'll start with 2 hidden layers, which use the [RELU]
(https://en.wikipedia.org/wiki/Rectifier_(neural_networks) activation function, which is a very 
simple rectifier function which essentially either returns x or zero. For our 
final output layer we will use a linear activation with matrix multiplication:
'''

def multilayer_perceptron(x,weights,biases):
    '''
    x:Placeholder for Data Input
    weights: Dictionary of weights
    biases: Dictionary of bias values
    '''
    
    # First hidden layer with RELU Activation
    # X * W + B
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    # RELU(X * W + B) -> f(x) = max(0,x)
    layer_1 = tf.nn.relu(layer_1)
    
    # Second hidden layer
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    # Last output layer
    out_layer = tf.matmul(layer_2,weights['out']) + biases['out']
    
    return out_layer

# Construct the model
pred = multilayer_perceptron(x,weights,bias)

'''
Cost and Optimization Function
'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


'''
next_batch()
Before we get started I want to cover one more convenience function in our mnist data object called next_batch. 
This returns a tuple in the form (X,y) with an array of the data and a y array indicating the class in the form of 
a binary array. For example:
'''
Xsamp,ysamp = mnist.train.next_batch(1)
plt.imshow(Xsamp.reshape(28,28))    
print(ysamp)    

'''
Run the session
'''    
# Launch the session
sess = tf.InteractiveSession()
# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

# Training Epochs
# Essentially the max amount of loops possible before we stop
# May stop earlier if cost/loss limit was set
import datetime
start = datetime.datetime.now()
print("The Model training started at :",start)
for epoch in range(training_epochs):
    
    # Start cost = 0.0
    avg_cost = 0.0
    # Convert total number of batches to integer
    total_batch = int(n_samples/batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        
        # Feed dictionary for optimization and loss value
        # Returns a tuple, but we only need 'c' the cost
        # So we set an underscore as a "throwaway"
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_X, y: batch_y})
        
        # compute average loss
        avg_cost += c/total_batch
        
    print("Epoch : {} cost = {:.4f}".format(epoch+1,avg_cost))

print("Model has completed {} Epochs of Training".format(training_epochs))
end = datetime.datetime.now()
print("The time taken for training the model is :",(end - start))

'''
Model Evaluations

This is essentially just a check of predictions == y_test. In our case since we know the format of 
the labels is a 1 in an array of zeroes, we can compare argmax() location of that 1. 
Remember that y here is still that placeholder we created at the very beginning, 
we will perform a series of operations to get a Tensor that we can eventually fill in 
the test data for with an evaluation method. 
'''

# Test the model
correct_predictions = tf.equal(tf.argmax(pred,axis=1),tf.argmax(y, axis=1))

# Correct_predictions is a tensor object
print(correct_predictions)

# In order to get a numerical value for our predictions we will need to use tf.cast to cast the Tensor 
# of booleans back into a Tensor of Floating point values in order to take the mean of it.

correct_predictions = tf.cast(correct_predictions,"float")

# Now we use the tf.reduce_mean function in order to grab the mean of the elements across the tensor
accuracy = tf.reduce_mean(correct_predictions)
type(accuracy)

'''
Now we can call the MNIST test labels and images and evaluate our accuracy!
'''
mnist.test.labels
mnist.test.images

# The eval() method allows you to directly evaluates this tensor in a Session without 
# needing to call tf.sess()

print("Accuracy: ",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))