"""
Demonstrates basic working of matplotlib visualizations
@author: suvosmac
"""
import numpy as np
import matplotlib.pyplot as plt

# Lets create a numpy array
x = np.linspace(0, 5, 11)
y = x ** 2

# There are two ways to create charts using matplotlib

# Functional method
plt.plot(x, y, 'b-')  # third argument is to add color
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.show()

# to add multiple plots on the viewing area
plt.subplot(1, 2, 1)  # divide the plotting area into 1 row and 2 columns and editing first plot
plt.plot(x, y, 'r')
plt.subplot(1, 2, 2)  # divide the plotting area into 1 row and 2 columns and editing the second plot
plt.plot(y, x, 'b')
plt.show()

# Object Oriented method
# The idea is to create a figure object and then call methods on it
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
axes.plot(x, y)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Set Title')
plt.show()

# Now to put two figures in one canvas
fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
# Now we will plot on these axes
axes1.plot(x, y)
axes1.set_title('Larger Plot')
axes2.plot(y, )
axes2.set_title('Smaller Plot')
plt.show()

# Lets divide the plotting area into 3 rows and 3 columns
fig, axes = plt.subplots(nrows=3, ncols=3)
# to fix the overlap
plt.tight_layout()
plt.show()

# for now lets just work with one row and 2 columns
fig, axes = plt.subplots(nrows=1, ncols=2)
# axes is an array which can be iterated and also index referenced
# for example we can say
axes[0].plot(x, y)
axes[0].set_title('First Plot')
axes[0].set_xlabel('X(1)')
axes[0].set_ylabel('Y(1)')
axes[1].plot(y, x)
axes[1].set_title('Second Plot')
axes[1].set_xlabel('Y(2)')
axes[1].set_ylabel('X(2)')
plt.tight_layout()
plt.show()

'''
Now lets work on figure size, aspect ratio and DPI
To set the figure size in inches (8 inches by 2 inches)
and DPI of 200 we can use
'''
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 2), dpi=200)
axes[0].plot(x, y)
axes[1].plot(y, x)
plt.show()

# How to save a figure
fig.savefig('my_picture.png', dpi=200)

# Now we will plot two variables at a time
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, x**2, label='Squared')  # Label attribute will be referenced by legend
ax.plot(x, x**3, label='Cubed')
ax.legend(loc=0)  # loc = 0, allows matplotlib to chose the best location for legend
plt.show()

# Setting colors and other cosmetic styles to matplotlib
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
# alpha argument controls the transparency
ax.plot(x, y, color="green", linewidth=2, alpha=0.5,
        linestyle="-", marker='o', markersize=10, markerfacecolor='black',
        markeredgecolor='red')
plt.show()

# Now to show just a part of the plot
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, y, color='purple', linewidth=2, linestyle='--')
# to show the plot between 0 and 1
ax.set_xlim([0, 1])
ax.set_ylim([0, 2])
plt.show()

# To plot a scatter plot
plt.scatter(x, y)
plt.show()