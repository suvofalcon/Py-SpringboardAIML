"""

Implement a basic tensor flow basic approach towards the MNIST dataset

Thie basic approach will implement the activation function as softmax regression

@author: suvosmac
"""
#

import matplotlib.pyplot as plt
import tensorflow as tf

# Import the MNIST datasets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# Check the type of the mnist tensor object
type(mnist)

# check the images in the train dataset of mnist
mnist.train.images

# how many training and test examples
mnist.train.num_examples
mnist.test.num_examples

# To visualize the data
mnist.train.images.shape

# To check the first image
mnist.train.images[0] # This is an array of 784 elements

# each of these images are 28 pixels by 28 pixels, so reshaping it
mnist.train.images[0].reshape(28,28)

# Now we will plot this image in greyscale instead of color
plt.imshow(mnist.train.images[0].reshape(28,28),cmap='gist_gray')

# This data is already normalized - all the values are rsnging between 0 and 1 and hence are probabilities,
# suitable for softmax regression approach

'''
Building the model

- Declare Placeholders
- Declare Variables
- Create Graph Operations
- Loss Function
- Optimizer
- Create session and run
'''
# data
x = tf.placeholder(tf.float32,shape=[None,784])

# Variables - weights and bias - to keep things simple, we are initializing it with zeros, just for simplicity
# 10 classes to be predicted (0 to 9)
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Create Graph Operations
y = tf.matmul(x,W) + b

# Loss function
y_true = tf.placeholder(tf.float32,shape=[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# Create session
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)

    for step in range(1000):

        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x:batch_x,y_true:batch_y})


    # Evaluate our model
    # we are asking to return the index where we have the highest softmax value (probability)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    # correct_prediction is a list of boolean. So converting it to floating point numbers
    # Then taking the average of the same for accuracy
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))