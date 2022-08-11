#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:39:12 2018

This python file shows the basics of tensor flow usage

@author: suvosmac
"""

# import tensorflow
import tensorflow as tf

# to check the tensorflow version being used
print("Tensor Flow version is:",tf.__version__)

# constants in tensorflow would be stored in a tensor object
hello = tf.constant("Hello World!!")
hello # hello is a tensor object with data type as string
type(hello)

# Now we will create tensor flow sessions. THis is a class for running tensor flow operations
# A session object encapsulates an environment, in which operation objects are executed
sess = tf.Session() 

# Now we will run the tensor object created earlier under this session
sess.run(hello)

# Check the type
type(sess.run(hello))

# Operations - Multiple operations can be run under a tensor flow session
x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print("Operations with Constants")
    print('Addition :', sess.run(x + y))
    print('Subtraction :', sess.run(x - y))
    print('Multiplication :', sess.run(x * y))
    print('Division :', sess.run(x / y))
    
# Another object type in tensor flow is called placeholder which can accepts value. This is needed as sometimes
# we may not have a constant right away

x = tf.placeholder(tf.int64)
y = tf.placeholder(tf.int64)

add = tf.add(x,y)
sub = tf.subtract(x,y)
mult = tf.multiply(x,y)

with tf.Session() as sess:
    print("Operation with Placeholders")
    print('addition :',sess.run(add,feed_dict = {x:20, y:30}))
    print('subtraction :',sess.run(sub,feed_dict = {x:20, y:30}))
    print('Multiplication :',sess.run(mult,feed_dict = {x:20, y:30}))
    
# Now we will show matrix multiplication with tensorflow

'''
The number of columns of the 1st matrix must equal the number of rows of the 2nd matrix.
And the result will have the same number of rows as the 1st matrix, and the same number of columns as 
the 2nd matrix.
'''

import numpy as np

a = np.array([[5.0,5.0]]) # 1 X 2 matrix
b = np.array([[2.0],[2.0]]) # 2 X 1 matrix

mat1 = tf.constant(a)
mat2 = tf.constant(b)

# to check the dimensions of the matrix
mat1.get_shape()
mat2.get_shape()

matrix_multi = tf.matmul(mat1, mat2)

with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)

const = tf.constant(10)
# to fill a 4X4 matrix with values 10 and some other basic operations
fill_mat = tf.fill((4,4),10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))
# fill up a 4x4 from a random normal distribution
myrandn = tf.random_normal((4,4),mean=0,stddev=1.0)
# random uniform distribution
myrandu = tf.random_uniform((4,4),minval=0,maxval=1)

# create a list of all these values
my_ops = [const,fill_mat,myzeros,myones,myrandn,myrandu]

with tf.Session() as sess:
    for index in my_ops:
        print(sess.run(index))
        print("\n")

'''
Tensor Flow graphs
Graphs are sets of connected nodes
The connections are referred to as edges
In Tensor flow each node is an operation with possible inputs that can supply some output

We will construct a graph and then execute it
'''

# A graph with two input nodes (n1,n2) and connected to one output node (n3)
n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1 + n2
with tf.Session() as sess:
    result = sess.run(n3)
print(result)

# Whenever tensorflow session is created, it also creates a default graph
print(tf.get_default_graph())

# An empty graph object can also 
g = tf.Graph()
print(g)

# to set g as the default graph
with g.as_default():
    print(g is tf.get_default_graph())

'''
There are two main types of tensor objects in a tensorflow graph - Variables and Placeholders

During the optimization process Tensor flow tunes the parameters of the model
Variables can hold the values of weights and biases throughout the session
Variables need to be initialized

Placeholders are initially empty and are used to feed in the actual training examples.
However they do need to be decalared a expected data type (tf.float32) with an optional shape argument
'''

my_tensor = tf.random_uniform((4,4),minval=0,maxval=1)
# create a variable
my_variable = tf.Variable(initial_value=my_tensor)

# Before we print any variable, we will go ahead and initialize all variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(my_variable))

# to declare a placeholder (Shape is an optional argument)
ph = tf.placeholder(tf.float32,shape=(4,4))