"""
This file will demonstrate various features and capabilities of python numpy package

@Author : suvosmac
"""

import numpy as np

# Different ways to create numpy arrays
# From List
mylist = [1, 2, 3]
print(type(mylist))

arr = np.array(mylist)
print(type(arr))

# Lets cast a nested list to numpy array
mylist = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(mylist)

np.array(mylist)  # This creates a two dimensional array/matrix
mymatrix = np.array(mylist)
print(mymatrix)
print(mymatrix.shape)

print("An Array of Sequential Numbers: \n")
print(np.arange(0, 10))  # start is inclusive and end is exclusive

print("An Array of Zeros: \n")
print(np.zeros(5))

print("A two dimensional array of Zeros : \n")
print(np.zeros((4, 10)))

print("An Array of Ones: \n")
print(np.ones(5))

# We can perform arithmetic operations on an array
print(np.ones((4, 4)) + 5)
print(np.ones((4, 4)) * 100)

# We can also create some specialised arrays using numpy

np.linspace(0,10,20)  # linearly spaced arrays
np.eye(5)  # 5 X 5 identity matrix

# random sample over a uniform distribution, between 0 and 1, this follows a uniform distribution
# every number has uniform probabilty of getting chosen
np.random.rand(4)
np.random.rand(5, 5)

# Standard normal distribution, normally distributed around zero
# This has a mean of 0 and standard deviation of 1
np.random.randn(3)

np.random.randint(0, 10)  # returns a random integer within the range

np.random.randint(1, 100, 10)  # 10 random integers between 0 and 100

# to get consistent results
np.random.seed(42)
np.random.rand(4)

ranarr = np.random.randint(0, 50, 10)
print(ranarr)

'''
Some useful attributes of Numpy arrays
'''
# create a 3 x 3 matrix with values ranging from 0 to 8
np.arange(0, 9).reshape(3, 3)

arr = np.arange(25)
arr.reshape(5, 5)

# to get max and min values from an array
print(ranarr.max())
print(ranarr.min())

# to get the argument (index of the max and min values)
print(ranarr.argmax())
print(ranarr.argmin())

'''
Numpy indexing and selection
'''
arr = np.arange(0, 11)
# to get the number in index location 8
print(arr[8])
print(arr[1:5]) # From index 1 to 5
print(arr[:5])  # From index 0 (beginning) to 5
# from index 5 , all the way to end
print(arr[5:])

# Arithmetic operations
print(arr + 100)
print(arr/2)

# slice of an array will always point to the original array
slice_arr = arr[:6]
print(slice_arr)
# Now we make a modification, the original array, respective index numbers are also affected
slice_arr[:] = 99
print(slice_arr)
print("The original array \n")
print(arr)

# If we do not want the original number to be affected, we should
# state an explicit copy
arr_copy = arr.copy()
arr_copy[:] = 10000
print(arr_copy)
print("The original array \n")
print(arr)

'''
Indexing on 2d array
'''
arr_2d = np.array([[5, 10, 15], [20, 25, 30], [30, 35, 40]])
print(arr_2d)
print(arr_2d.shape)

# to get the first row and first column element
print(arr_2d[1][1])

# 2d array slicing - first two rows and second column onwards
print(arr_2d[:2, 1:])
# one more 2d array slicing
print(arr_2d[1:, :2])

'''
Conditional Selection
'''
arr = np.arange(1, 11)
# to get all elements greater than 4
print(arr > 4)
bool_arr = arr > 4

# now we will do conditional selection by passing boolean array into the original array
# will only return back array elements for which the boolean conditions are true
print(arr[bool_arr])
# The short way of doing it is
print(arr[arr > 4])
print(arr[arr <= 6])

'''
Numpy operations
'''
arr = np.arange(0, 10)
print(arr + 100)
print(arr / 100)
print((arr + 2)/100)
print(arr + arr)  # we can also do this arithmetic operations on two arrays of the same shape

print(1/arr)
print(arr/arr)
print(np.sqrt(arr))
print(np.log(arr))
print(np.sin(arr))
print(arr.sum())
print(arr.mean())

arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(arr_2d.shape)
print(arr_2d.sum())

# if we do the sum of all the rows - give me the sum across the rows
print(arr_2d.sum(axis=0))
print(arr_2d.sum(axis=1))  # across the columns
