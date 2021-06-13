'''
Demo for Broadcasting
'''

import numpy as np

# Define an arbitary array
# Calories from Carbs, Proteins, Fats in 100g of different foods
A = np.array([[56.0, 0.0, 4.4, 68.0],
            [1.2, 104.0, 52.0, 8.0],
            [1.8, 135.0, 99.0, 0.9]])

print(A)

# Calculate the sum of elements by column
cal = A.sum(axis=0)
print(cal)
print(f"Shape of the cal matrix {cal.shape}")

print(f"Percentage of Calories across brands of foods")
percentage = 100*A/cal # Here we see (3,4) matrix gets divided by (4,1) matrix
print(percentage)
print("\n")

'''
Some important tips
'''

a = np.random.rand(5)
print(f"This creates a array with dimension - {a.shape} - should be avoided in deep learnintg")
a = np.random.rand(5, 1)
print(f"This creates a array with dimension - {a.shape} - column Vector")
print(a)
a = np.random.rand(1, 5)
print(f"This creates a array with dimension - {a.shape} - row Vector")
print(a)