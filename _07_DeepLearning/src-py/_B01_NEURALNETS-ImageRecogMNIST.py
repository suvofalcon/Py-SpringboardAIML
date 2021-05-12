# -*- coding: utf-8 -*-
"""
Building a neural network for Image Recognition With MNIST Data
Dataset consists of images of handwritten digits

@author: suvosmac
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("//Volumes/Data/CodeMagic/Data Files/Jigsaw/mnist_train.csv",
                   header = None)
# Lets check the dataframe dimensions

print("The Dimensions", data.shape)
print("Head of the data")
"""
The first column is the label of the image and remaining 784 columns are
the pixel values of the corresponding grey scale images
"""
data.head()

"""
The task is to build a machine learning classifier that will be able to 
predict the digits from new and unlabelled images
This is a multi class classification problem with 10 classes
"""

# Split into training and testing splits
# Calculate the size of each partition
numTrain = int(0.7 * data.shape[0])
print(numTrain)

# fix a seed
np.random.seed(123)
# randomly shuffle the original dataset
np.random.shuffle(data.values)

# create the train, test and validate split using slicing notation
trainData = data[:numTrain]
testData = data[numTrain:]

print ("Training data shape", trainData.shape)
print ("Testing data shape", testData.shape)

"""
Lets try to plot a sample image
"""
# We store the first row of the training split omitting the first column in a variable called sampleRow
sampleRow = trainData.loc[0,1:].values
np.shape(sampleRow)

# we will rearrange these values in 28X28 matrix as the images are of 28 pixels by 28 pixels
imageData = sampleRow.reshape(28,28)
print(imageData.shape)

from matplotlib.pyplot import imshow

imshow(imageData, cmap = 'Greys_r', interpolation = 'None' )
plt.title("label: " + str(trainData.loc[0,0]))
plt.show()

# If we see carefully most of the pixels in an image is always black,specially the corners.
# That is why there are many zeros in an image 

# Now we will use some feature reduction techniques
"""
From a statistical point of view, if the data array is constant , it has 0 variance.
Features that do not vary across the data gives no information, moreover they can hamper the performance
of some classifiers.
"""
trainX = trainData[trainData.columns[1:]]
# compute variances on each column
variances = trainX.apply(lambda x:np.var(x), axis = 0)

#Lets study the distribution of variances
variances.hist()
plt.show()
# The distribution shows quite a few columns have zero variances

# The columns which has non zero variances
trainX.columns[variances > 0]
# Lets view the summary statistics of the variance variances
variances.describe()

# We can get rid of these columns in our modelling

'''
Modelling the neural network
'''
from keras.models import Sequential
from keras.layers import Dense, Activation

# Seggregating between training features and response variable
trainX = trainData[trainData.columns[1:]]
trainY = trainData[trainData.columns[0]]

# One hot encoding of the target variable, because it is a multi class classification
from keras.utils import np_utils
encodedY = np_utils.to_categorical(trainY)
encodedY

# Initialize an empty model
model = Sequential()

# add first hidden layer
model.add(Dense(input_dim = trainX.shape[1], activation = 'sigmoid', units = 100,
                kernel_initializer = 'uniform', use_bias = True))

# add output layer
model.add(Dense(units = 10, kernel_initializer = 'uniform', 
                use_bias = True, activation = 'sigmoid'))

# Check the summary
model.summary()

# compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])

# fit the model
model.fit(trainX,encodedY,epochs=100,batch_size=10)

# Do the predictions
predictionsTestY = model.predict_classes(testData[testData.columns[1:]])
np.mean(predictionsTestY == testData[testData.columns[0]])

# We can try to improve the accuracy of this network by adding additional layer
"""
Tuning involves
- Adding additional hidden layers
- Tweaking the number of nodes in each of these layers
- Combine this with the Cross-Validation module in Scikit Learn
- to achieve robust results
"""

# To build and check the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(predictionsTestY,testData[testData.columns[0]])