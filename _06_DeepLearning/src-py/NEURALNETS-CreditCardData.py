#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 19:12:54 2018

@author: suvosmac
"""
'''
Building neural network on the credit card data
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the credit card data file
credit_data = pd.read_csv("//Volumes/Data/CodeMagic/Data Files/Jigsaw/Credit Default Data_demo session.csv")

# Some quick info on the dataset
credit_data.info() #29,601 data elements in 25 rows
# Check initial rows of the data
credit_data.head()
# to summarise the data
credit_data.describe()

# The ID variable is of no use, so lets drop the ID variable
credit_data = credit_data.drop("ID",axis=1)
# Convert the response column to categorical
credit_data['default payment next month'] = pd.Categorical(credit_data['default payment next month'])

# Lets explore the response variable a bit
print(credit_data['default payment next month'].value_counts())
tmpvar = credit_data['default payment next month'].value_counts()
print("Proportion of Defaulters %0.2f%%" %(tmpvar[1]/tmpvar.sum() * 100))
sns.countplot(x='default payment next month',data=credit_data)
# We see in the dataset, there are approximately 30% data for defaulters

# View other factor variables
print("Distribution of Sex")
print(credit_data['SEX'].value_counts())
sns.countplot(x='SEX',data=credit_data)

print("Distribution of Education")
print(credit_data['EDUCATION'].value_counts())
sns.countplot(x='EDUCATION',data=credit_data)

print("Distribution of Marital Status")
print(credit_data['MARRIAGE'].value_counts())
sns.countplot(x='MARRIAGE',data=credit_data)

tmpvar = credit_data['default payment next month'].value_counts()
print("Proportion of Defaulters %0.2f%%" %(tmpvar[1]/tmpvar.sum() * 100))
tmpvar = credit_data['SEX'].value_counts()
print("Proportion of Male Customers %0.2f%%" %(tmpvar[1]/tmpvar.sum() * 100))
tmpvar = credit_data['MARRIAGE'].value_counts()
print("Proportion of Married Customers %0.2f%%" %(tmpvar[1]/tmpvar.sum() * 100))
tmpvar = credit_data['EDUCATION'].value_counts()
print("Proportion of Graduate Customers %0.2f%%" %(tmpvar[1]/tmpvar.sum() * 100))

# Lets study relationship between categorical variables using crosstabs
print('Distribution of Sex and Credit Default')
print(pd.crosstab(credit_data['SEX'],credit_data['default payment next month']))
print('Proportional Distribution of Sex and Credit Default')
print(pd.crosstab(credit_data['SEX'],credit_data['default payment next month'])
    .apply(lambda r: r/r.sum(), axis = 1))
sns.countplot(x='default payment next month',data=credit_data,hue='SEX')

print('Distribution of Education and Credit Default')
print(pd.crosstab(credit_data['EDUCATION'],credit_data['default payment next month']))
print('Proportional Distribution of Education and Credit Default')
print(pd.crosstab(credit_data['EDUCATION'],credit_data['default payment next month'])
    .apply(lambda r: r/r.sum(), axis = 1))
sns.countplot(x='default payment next month',data=credit_data,hue='EDUCATION')

print('Distribution of Marriage and Credit Default')
print(pd.crosstab(credit_data['MARRIAGE'],credit_data['default payment next month']))
print('Proportional Distribution of Marriage and Credit Default')
print(pd.crosstab(credit_data['MARRIAGE'],credit_data['default payment next month'])
    .apply(lambda r: r/r.sum(), axis = 1))
sns.countplot(x='default payment next month',data=credit_data,hue='MARRIAGE')

# Lets study the age variable with respect to response
credit_data.groupby('default payment next month').describe()['AGE']
sns.barplot(x='default payment next month',y='AGE',data=credit_data)
sns.stripplot(x='default payment next month',y='AGE',data=credit_data,jitter=True)
sns.boxplot(x='default payment next month',y='AGE',data=credit_data)
# Distribution of Age by Defaulters and Non Defaulters
fig,axes = plt.subplots(nrows=1, ncols=2)
sns.distplot(credit_data[credit_data['default payment next month'] == 0]['AGE'],ax=axes[0])
axes[0].set_title('Default = 0')
sns.distplot(credit_data[credit_data['default payment next month'] == 1]['AGE'],ax=axes[1])
axes[1].set_title('Default = 1')
plt.tight_layout()

# Lets study the Limit Balance Column with respect to the response variable
credit_data.groupby('default payment next month').describe()['LIMIT_BAL']
sns.barplot(x='default payment next month',y='LIMIT_BAL',data=credit_data)
sns.stripplot(x='default payment next month',y='LIMIT_BAL',data=credit_data,jitter=True)
sns.boxplot(x='default payment next month',y='LIMIT_BAL',data=credit_data)
# Distribution of LIMIT_BAL by Defaulters and Non Defaulters
fig,axes = plt.subplots(nrows=1, ncols=2)
sns.distplot(credit_data[credit_data['default payment next month'] == 0]['LIMIT_BAL'],ax=axes[0])
axes[0].set_title('Default = 0')
sns.distplot(credit_data[credit_data['default payment next month'] == 1]['LIMIT_BAL'],ax=axes[1])
axes[1].set_title('Default = 1')
plt.tight_layout()
# Shows people who have defaulted have lower average credit balance

# Since the numerical variables have different scales, we will perform variable standardization
# Perform variable standardization
numericColList = ['LIMIT_BAL','AGE','BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
credit_data_std = credit_data[numericColList].apply(lambda x:(x - np.mean(x))/np.std(x))
credit_data_std.describe()
# now we will add the response column from the original dataframe
credit_data_std['default'] = credit_data['default payment next month']
# Now the values are standardized, we can compare two boxplots with respect to the response variable.
# For example box plot for AGE and Box plot for LIMIT_BAL as compared to response variable
fig,axes = plt.subplots(nrows=1, ncols=2)
sns.boxplot(x='default',y='AGE',data = credit_data_std,ax=axes[0])
axes[0].set_title('AGE')
sns.boxplot(x='default',y='LIMIT_BAL',data = credit_data_std,ax=axes[1])
axes[1].set_title('LIMIT_BAL')
plt.tight_layout()

# Plotting all boxplots one after another
fig,axes = plt.subplots(nrows=4, ncols=2)
sns.boxplot(x='default',y='AGE',data = credit_data_std,ax=axes[0,0])
axes[0,0].set_title('AGE')
sns.boxplot(x='default',y='LIMIT_BAL',data = credit_data_std,ax=axes[0,1])
axes[0,1].set_title('LIMIT_BAL')
sns.boxplot(x='default',y='BILL_AMT1',data = credit_data_std,ax=axes[1,0])
axes[1,0].set_title('BILL_AMT1')
sns.boxplot(x='default',y='BILL_AMT2',data = credit_data_std,ax=axes[1,1])
axes[1,1].set_title('BILL_AMT3')
sns.boxplot(x='default',y='BILL_AMT4',data = credit_data_std,ax=axes[2,0])
axes[2,0].set_title('BILL_AMT4')
sns.boxplot(x='default',y='BILL_AMT5',data = credit_data_std,ax=axes[2,1])
axes[2,1].set_title('BILL_AMT5')
sns.boxplot(x='default',y='BILL_AMT6',data = credit_data_std,ax=axes[3,0])
axes[3,0].set_title('BILL_AMT6')
sns.boxplot(x='default',y='PAY_AMT1',data = credit_data_std,ax=axes[3,1])
axes[3,1].set_title('PAY_AMT1')
sns.boxplot(x='default',y='PAY_AMT2',data = credit_data_std,ax=axes[4,0])
axes[4,0].set_title('PAY_AMT2')
sns.boxplot(x='default',y='PAY_AMT3',data = credit_data_std,ax=axes[4,1])
axes[4,1].set_title('PAY_AMT3')
plt.tight_layout()
# Variable standardization makes the variable unit free and this makes comparison of two variable graphs possible

# Study relationship between AGE and LIMIT_BAL by Default using scatter plot
sns.jointplot(x='AGE',y='LIMIT_BAL',data=credit_data_std)
sns.pairplot(credit_data_std,vars=['AGE','LIMIT_BAL'],hue='default')
# to study relationship between all numerical variables by the defaulter
sns.pairplot(credit_data_std,hue='default')
# We see bill amounts are highly correlated. High bill amounts in a month means higher bill
# amount in the next month as well

# Since all bill amounts are highly correlated, instead of having six bill amount variables, we
# can have one average variable for all of them
AVG_BILL_AMOUNT = credit_data[numericColList[2:8]].apply(lambda x:np.mean(x),axis=1)
# now we will standardize this variable and put in our standardized dataframe
credit_data_std['AVG_BILL_AMIT'] = (AVG_BILL_AMOUNT - np.mean(AVG_BILL_AMOUNT)/np.std(AVG_BILL_AMOUNT))
# Lets vidualise this data
sns.barplot(x='default',y='AVG_BILL_AMIT',data=credit_data_std)
sns.boxplot(x='default',y='AVG_BILL_AMIT',data=credit_data_std)
# There seems to be no relationship between average bill amount and being a defaulter

"""
We will now partition the data into train, test and validate
"""
numTrain, numTest = int(0.7 * credit_data.shape[0]), int(0.2 * credit_data.shape[0])
# Randomly shuffle the original dataset
np.random.shuffle(credit_data.values)

# We will create the train, test and validate split using slicing notation
trainData = credit_data[:numTrain]
testData = credit_data[numTrain:(numTrain+numTest)]
validateData = credit_data[(numTrain+numTest):]
print('Dimensions of Training data %s' %(trainData.shape,))
print('Dimensions of Test data %s' %(testData.shape,))
print('Dimensions of Validate data %s' %(validateData.shape,))

# create groups of variables : response, factor and numeric variables
response = 'default payment next month'
factorVars = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

allCols = list(credit_data.columns.values)

# another way to create the list of numeric variables, using list comprehension
numericVars = []
[numericVars.append(i) for i in allCols if i not in factorVars + [response]]
print(numericVars)

# Now we will do variable standaridization
from sklearn.preprocessing import StandardScaler
# create an instance of StandardScaler
trainX = trainData[numericVars].values
scaler = StandardScaler()
scaler.fit(trainX)

print ('List of means using the training data :', scaler.mean_)
print ('List of standard deviation using the training data :', scaler.scale_)
trainX_std = scaler.transform(trainX)
print(trainX_std.shape)
print(trainX_std)

from sklearn.feature_extraction import DictVectorizer
dv = (DictVectorizer(sparse = False).fit(trainData[factorVars].applymap(str).T.to_dict().values()))
catXTrainEnc = dv.transform(trainData[factorVars].applymap(str).T.to_dict().values())
print(catXTrainEnc.shape)
print(catXTrainEnc)

# Lets join everything together
trainX_transformed = np.hstack((trainX_std,catXTrainEnc))
print("Training data shape after transformation", trainX_transformed.shape)

# take the response values from the training set
trainY = trainData[response].values

print("feature set for training")
trainX_transformed
print("Response for Training")
trainY

'''
Now we will implement the preprocessing pipeline as a function
'''
def preProcessing(dataFrame, scalarObj, encodingObj):
    
    '''
    input:
        dataFrame: The dataframe for which preprocessing needs to be done
        scalarObj: The object returned by preprocessing. scalar.fit
        encodingObj: The object returned after encoding categorical variable
    output:
        2 objects:
            transformed set of features
            response
    '''
    
    NumX = dataFrame[numericVars]
    CatX = dataFrame[factorVars]
    Y = dataFrame[response]
    
    # Standardize the numeric variable
    NumX_std = scalarObj.transform(NumX)
    
    #encode the categorical variables
    CatX_enc = encodingObj.transform(CatX.applymap(str).T.to_dict().values())
    
    #create the transformed training set
    X_transformed = np.hstack((NumX_std,CatX_enc))
    Y = dataFrame[response].values
    return[X_transformed,Y]

trainX_transformed, trainY = preProcessing(trainData,scaler,dv)
testX_transformed, testY = preProcessing(testData,scaler,dv)    
validateX_transformed, validateY = preProcessing(validateData,scaler,dv)    

print ("Transformed Training Split")
trainX_transformed
print(trainX_transformed.shape)
print ("Transformed Testing Split")    
testX_transformed
print("Transformed Validation Split")
validateX_transformed

'''
To start with, we will create a simple architecture for the neural network
'''
from keras.models import Sequential
from keras.layers import Dense, Activation

# initialize an empty model
model = Sequential()

inputDim = trainX_transformed.shape[1]

# Add the first hidden layer
# The number of neurons in the hidden layer, will be half of the input layer
'''
The choice of number of neurons in the hidden layer is quite arbitary and we will deal
with hyperparameters later.
We initialize the weights using uniform distribution. The activation function is sigmoid as this
this is a classification problem and we decide use a bias 
'''
model.add(Dense(input_dim = inputDim, activation = 'sigmoid', units = int(inputDim/2),
                kernel_initializer = 'uniform', use_bias = True))
# Add the output layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', 
                use_bias = True, activation = 'sigmoid'))
# view the summary
model.summary()

# Now we will have to compile the model, we created
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])
# Since we are using a binary classifier, we change our cost function to binary crossentropy.
# It is also called the Log loss function

# Now we will train the model
import datetime
start = datetime.datetime.now()
print('The training begins : ',start)
model.fit(trainX_transformed, trainY, nb_epoch = 50, batch_size = 10)
end = datetime.datetime.now()
print("The time taken for training the model is :",(end - start))

# Now we will use the model to predict on the testing split
predictionsTestY = model.predict_classes(testX_transformed)
"""
The evaluate model, returns a list of length 2. The first element of the list is the value of
the loss function evaluated on the testing split using the final weights of the trained model
The second element is the accuracy of the predictions on the testing split
"""
model.evaluate(testX_transformed, testY)

"""
Hyperparamter optimization
Hyper parameters for neural networks cannot be estimated directly from backpropagation
algorithm. Every option encountered while creating keras models, can be treated as 
hyperparameter whose value can be optimised from the training data

These options include
- Learning rate ETA for Gradient Descent Algorithm
- Number of hidden layers in the neural network
- Number of neurons in the hidden layer
- Batch size required for updating the weights
- Number of epochs cotrolling iterations for the gradient descent algorithm
- Regularization parameter for constraining the weight of the neural networks
"""

# Now we illustrate a relatively simple grid search for two hyperparameters
numHiddenLayers = [1,2]
epochs = [10,25,50]

import itertools
grid2D = [element for element in itertools.product(numHiddenLayers,epochs)]
print(grid2D)

# The objective now is to train different neural nets as per the combinations in the
# grid2D and to gather accuracies for the corresponding models on the testing split

# Based on the results we will make our final selection

# create a model with one hidden layer
# initialize an empty model
model1Hidden = Sequential()
inputDim = trainX_transformed.shape[1]

# add first hidden layer
model1Hidden.add(Dense(input_dim = inputDim, activation = 'sigmoid', units = int(inputDim/2),
                       kernel_initializer = 'uniform', use_bias = True))

# add output layer
model1Hidden.add(Dense(units = 1, kernel_initializer = 'uniform', use_bias = True,
                       activation = 'sigmoid'))

# Create a model with two hidden layers
# initialize an empty model
model2Hidden = Sequential()
inputDim = trainX_transformed.shape[1]

# add first hidden layer
model2Hidden.add(Dense(input_dim = inputDim, activation = 'sigmoid', units = int(inputDim/2),
                       kernel_initializer = 'uniform', use_bias = True))
# add the second hidden layer
model2Hidden.add(Dense(int(inputDim/2), kernel_initializer = 'uniform', use_bias = True,
                       activation = 'sigmoid'))
# add the output layer
model2Hidden.add(Dense(units = 1, kernel_initializer = 'uniform', use_bias = True,
                       activation = 'sigmoid'))

# Now we will compile both the models
model1Hidden.compile(loss = 'binary_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])
model2Hidden.compile(loss = 'binary_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])

# Now we will run the two models, for each of the six combinations
accuracyList = []
start = datetime.datetime.now()
print('The training begins : ',start)
for eachCombination in grid2D:
    print("For Combination",eachCombination)
    if eachCombination[0] == 1:
        #train the model with 1 hidden layer
        model1Hidden.fit(trainX_transformed, trainY,
                         epochs = eachCombination[1],
                         batch_size = 10)
        accuracyList.append(model1Hidden.evaluate(testX_transformed,testY))
        
    else:
        model2Hidden.fit(trainX_transformed, trainY, epochs = eachCombination[1],
                         batch_size = 10)
        accuracyList.append(model1Hidden.evaluate(testX_transformed,testY))

print(list(zip(grid2D, accuracyList))) 
end = datetime.datetime.now()
print("The time taken for training the model is :",(end - start))

# From the list, we see the model with 1 hidden layer and epoch value of 25
# performs the best

# We will now, use the same parameters to construct the model the run it on validation set
model1Hidden.fit(trainX_transformed,trainY,epochs = 25, batch_size = 10)
print("Running on Test Data")
scores = model1Hidden.evaluate(testX_transformed,testY)
print("Accuracy on Test Data %0.2f" %(scores[1]*100))
print("Running on Validation Data")
scores = model1Hidden.evaluate(validateX_transformed,validateY)      
print("Accuracy on Validation Data %0.2f" %(scores[1]*100))

"""
Now we will integrate between scikit learn and keras
"""
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold

# function to create model, required for KerasClassifier
def nnModel():
    #recreate the optimum model
    model = Sequential()
    model.add(Dense(input_dim = inputDim, activation = 'sigmoid', units = int(inputDim/2),
                       kernel_initializer = 'uniform', use_bias = True))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', use_bias = True,
                       activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])
    return model

# now we will create this model, so that we can use with sklearn
#modelForSklearn = KerasClassifier(build_fn = nnModel, epochs = 25, batch_size = 10)

# Lets fix a random seed
seed = 7
np.random.seed(seed)
# setup a 5-fold cross validation
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
cvscores = []
for train, test in folds.split(trainX_transformed, trainY):
    model = nnModel()
    #fit the model
    model.fit(trainX_transformed[train],trainY[train],epochs = 25, batch_size = 10)
    #evaluate the model
    scores = model.evaluate(trainX_transformed[test],trainY[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))