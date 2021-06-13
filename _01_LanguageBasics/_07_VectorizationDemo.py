'''
we will demonstrate how a vectorized version is faster than regular forloops
'''

import numpy as np
import time

# Two arbirary arrays
a = np.random.rand(1000000)
b = np.random.rand(1000000)

print("Array - a")
print(a)
print("Array - b")
print(b)

# Performing vector product
tic = time.time()
vecProduct = np.dot(a, b)
toc = time.time()

print(f"Vector Product - {vecProduct}")
print(f"Vector Product time - {1000 * (toc-tic)} ms")

# Performing Normal For loop multiplication
tic = time.time()
normProduct = 0
for index in range(1000000):
    normProduct += a[index] * b[index]
toc = time.time()

print(f"For Loop Product - {normProduct}")
print(f"For Loop Product time - {1000 * (toc-tic)} ms")