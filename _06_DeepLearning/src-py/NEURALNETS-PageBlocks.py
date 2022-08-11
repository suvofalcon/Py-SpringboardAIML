"""
Page Blocks Classification Introduction
Processing paper documents is an important tasks in office automation. It involves much more than 
just a simple acquisition of a paper document by means of a scanner, Generally speaking, 
a paper document is a collection of printed objects (characters, columns, paragraphs, titles, 
figures, and so on), each of which needs to be detected and then processed in the most 
appropriate way. A segmentation process can quite accurately identify the different blocks 
in a document but it cannot identify them, i.e. it is not possible for the segmentation 
process to tell if a block is text or an image.
The problem is then to classify all the blocks of the page layout of a document that has been 
detected by a segmentation process. All blocks are manually labelled as image, text etc. 
on a historical dataset. The task is to use this historical dataset to predict the type of 
information contained in a block using artificial neural networks.

Data Description
The dataset contains information on 5,473 blocks identified from 54 documents.
1. height: height of the block.
2. length: length of the block.
3. area: Area of the block (height * length);
4. eccen: Eccentricity of the block (length / height);
5. p_black: Percentage of black pixels within the block (blackpix / area);
6. p_and: Percentage of black pixels after the application of the Run Length Smoothing 
Algorithm (RLSA) (blackand / area);
7. mean_tr: Mean number of white-black transitions (blackpix / wb_trans); 
8. blackpix: Total number of black pixels in the original bitmap of the block. 
9. blackand: Total number of black pixels in the bitmap of the block after the RLSA.
10. wb_trans: Number of white-black transitions in the original bitmap of the block.
11. Class: the class of the block. The five classes are: text (1), horizontal line (2), 
picture (3), vertical line (4) and graphic (5).

Task:
In the data folder, you are given 2 csv files train and test. We have split the original data i
n 2 partitions. Train your neural network on the train data set and evaluate the final model 
on the test split.

@author: suvosmac
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# We will now load the dataset
pageBlocks_train = pd.read_csv('//Volumes/Data/CodeMagic/Data Files/Jigsaw/page-blocks_train.csv')
# Some quick info on the dataset
pageBlocks_train.describe()
pageBlocks_train.info()
# examine the initial rows
pageBlocks_train.head()

# categories of the response variable
pageBlocks_train['CLASS'].unique()

# The response variable currently integer has to converted as category
#pageBlocks_train['CLASS'] = pageBlocks_train['CLASS'].astype('category')


'''
Now we perform exploratory data analysis
'''
# Explore the response variable
print(pageBlocks_train['CLASS'].value_counts())
tmpvar = pageBlocks_train['CLASS'].value_counts()
print("Proportion of Text %0.2f%%" %(tmpvar[1]/tmpvar.sum() * 100))
print("Proportion of Horizontal Line %0.2f%%" %(tmpvar[2]/tmpvar.sum() * 100))
print("Proportion of Picture %0.2f%%" %(tmpvar[3]/tmpvar.sum() * 100))
print("Proportion of Vertical Line %0.2f%%" %(tmpvar[4]/tmpvar.sum() * 100))
print("Proportion of Graphic %0.2f%%" %(tmpvar[5]/tmpvar.sum() * 100))
sns.countplot(x = 'CLASS',data = pageBlocks_train)

def featureDistributuionByResponse(feature,response):
    fig,axes = plt.subplots(nrows=1, ncols=5)
    sns.distplot(pageBlocks_train[pageBlocks_train[response] == 1][feature],ax=axes[0])
    axes[0].set_title('CLASS = 1')
    sns.distplot(pageBlocks_train[pageBlocks_train[response] == 2][feature],ax=axes[1])
    axes[1].set_title('CLASS = 2')
    sns.distplot(pageBlocks_train[pageBlocks_train[response] == 3][feature],ax=axes[2])
    axes[2].set_title('CLASS = 3')
    sns.distplot(pageBlocks_train[pageBlocks_train[response] == 4][feature],ax=axes[3])
    axes[3].set_title('CLASS = 4')
    sns.distplot(pageBlocks_train[pageBlocks_train[response] == 5][feature],ax=axes[4])
    axes[4].set_title('CLASS = 5')
    plt.tight_layout()


# Explore relationship with HEIGHT
pageBlocks_train.groupby('CLASS').describe()['HEIGHT']
sns.barplot(x='CLASS',y='HEIGHT',data=pageBlocks_train)
sns.stripplot(x='CLASS',y='HEIGHT',data=pageBlocks_train,jitter=True)
sns.boxplot(x='CLASS',y='HEIGHT',data=pageBlocks_train)
# Distribution of HEIGHT by CLASS
featureDistributuionByResponse('HEIGHT','CLASS')

# Explore relationship with LENGTH
pageBlocks_train.groupby('CLASS').describe()['LENGTH']
sns.barplot(x='CLASS',y='LENGTH',data=pageBlocks_train)
sns.stripplot(x='CLASS',y='LENGTH',data=pageBlocks_train,jitter=True)
sns.boxplot(x='CLASS',y='LENGTH',data=pageBlocks_train)
# Distribution of LENGTH by CLASS
featureDistributuionByResponse('LENGTH','CLASS')

# Explore relationship with AREA
pageBlocks_train.groupby('CLASS').describe()['AREA']
sns.barplot(x='CLASS',y='AREA',data=pageBlocks_train)
sns.stripplot(x='CLASS',y='AREA',data=pageBlocks_train,jitter=True)
sns.boxplot(x='CLASS',y='AREA',data=pageBlocks_train)
# Distribution of AREA by CLASS
featureDistributuionByResponse('AREA','CLASS')

# Explore relationship with ECCEN
pageBlocks_train.groupby('CLASS').describe()['ECCEN']
sns.barplot(x='CLASS',y='ECCEN',data=pageBlocks_train)
sns.stripplot(x='CLASS',y='ECCEN',data=pageBlocks_train,jitter=True)
sns.boxplot(x='CLASS',y='ECCEN',data=pageBlocks_train)
# Distribution of ECCEN by CLASS
featureDistributuionByResponse('ECCEN','CLASS')

# Explore relationship with P_BLACK
pageBlocks_train.groupby('CLASS').describe()['P_BLACK']
sns.barplot(x='CLASS',y='P_BLACK',data=pageBlocks_train)
sns.stripplot(x='CLASS',y='P_BLACK',data=pageBlocks_train,jitter=True)
sns.boxplot(x='CLASS',y='P_BLACK',data=pageBlocks_train)
# Distribution of P_BLACK by CLASS
featureDistributuionByResponse('P_BLACK','CLASS')

# Explore relationship with P_AND
pageBlocks_train.groupby('CLASS').describe()['P_AND']
sns.barplot(x='CLASS',y='P_AND',data=pageBlocks_train)
sns.stripplot(x='CLASS',y='P_AND',data=pageBlocks_train,jitter=True)
sns.boxplot(x='CLASS',y='P_AND',data=pageBlocks_train)
# Distribution of P_AND by CLASS
featureDistributuionByResponse('P_AND','CLASS')

# Explore relationship with MEAN_TR
pageBlocks_train.groupby('CLASS').describe()['MEAN_TR']
sns.barplot(x='CLASS',y='MEAN_TR',data=pageBlocks_train)
sns.stripplot(x='CLASS',y='MEAN_TR',data=pageBlocks_train,jitter=True)
sns.boxplot(x='CLASS',y='MEAN_TR',data=pageBlocks_train)
# Distribution of MEAN_TR by CLASS
featureDistributuionByResponse('MEAN_TR','CLASS')

# Explore relationship with BLACKPIX
pageBlocks_train.groupby('CLASS').describe()['BLACKPIX']
sns.barplot(x='CLASS',y='BLACKPIX',data=pageBlocks_train)
sns.stripplot(x='CLASS',y='BLACKPIX',data=pageBlocks_train,jitter=True)
sns.boxplot(x='CLASS',y='BLACKPIX',data=pageBlocks_train)
# Distribution of BLACKPIX by CLASS
featureDistributuionByResponse('BLACKPIX','CLASS')

# Explore relationship with BLACKAND
pageBlocks_train.groupby('CLASS').describe()['BLACKAND']
sns.barplot(x='CLASS',y='BLACKAND',data=pageBlocks_train)
sns.stripplot(x='CLASS',y='BLACKAND',data=pageBlocks_train,jitter=True)
sns.boxplot(x='CLASS',y='BLACKAND',data=pageBlocks_train)
# Distribution of BLACKAND by CLASS
featureDistributuionByResponse('BLACKAND','CLASS')

# Explore relationship with WB_TRANS
pageBlocks_train.groupby('CLASS').describe()['WB_TRANS']
sns.barplot(x='CLASS',y='WB_TRANS',data=pageBlocks_train)
sns.stripplot(x='CLASS',y='WB_TRANS',data=pageBlocks_train,jitter=True)
sns.boxplot(x='CLASS',y='WB_TRANS',data=pageBlocks_train)
# Distribution of WB_TRANS by CLASS
featureDistributuionByResponse('WB_TRANS','CLASS')

'''
Neural Network Modelling
'''
# Isolate the features and response variable
X = pageBlocks_train.drop(columns='CLASS',axis = 1)
y = pageBlocks_train['CLASS']

# Let us standardize the feature dataframe
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Divide this into train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=101)

# One hot encoding of the target variable in train and test, because it is a multi class classification
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y_train)
encoder.fit(y_test)
encodedY = encoder.transform(y_train)
encodedY_test = encoder.transform(y_test)
encodedY = np_utils.to_categorical(encodedY)
encodedY_test = np_utils.to_categorical(encodedY_test)

# Now we illustrate a relatively simple grid search for two hyperparameters
numHiddenLayers = [1,2]
epochs = [10,25,50,100]

import itertools
grid2D = [element for element in itertools.product(numHiddenLayers,epochs)]
print(grid2D)

'''
The objective now is to train different neural nets as per the combinations in the
grid2D and to gather accuracies for the corresponding models on the testing split

Based on the results we will make our final selection
'''

# Build various model based on different hyperparameter
from keras.models import Sequential
from keras.layers import Dense, Activation

inputDim = X_train.shape[1]

# create a model with one hidden layer
# initialize an empty model
model1Hidden = Sequential()
# add first hidden layer
model1Hidden.add(Dense(input_dim = inputDim, activation = 'sigmoid', units = int(inputDim/2),
                       kernel_initializer = 'uniform', use_bias = True))

# add output layer
model1Hidden.add(Dense(units = pageBlocks_train['CLASS'].nunique(), kernel_initializer = 'uniform', use_bias = True,
                       activation = 'sigmoid'))

# Create a model with two hidden layers
# initialize an empty model
model2Hidden = Sequential()

# add first hidden layer
model2Hidden.add(Dense(input_dim = inputDim, activation = 'sigmoid', units = int(inputDim/2),
                       kernel_initializer = 'uniform', use_bias = True))
# add the second hidden layer
model2Hidden.add(Dense(int(inputDim/2), kernel_initializer = 'uniform', use_bias = True,
                       activation = 'sigmoid'))
# add the output layer
model2Hidden.add(Dense(units = pageBlocks_train['CLASS'].nunique(), kernel_initializer = 'uniform', use_bias = True,
                       activation = 'sigmoid'))

# Now we will compile both the models
model1Hidden.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])
model2Hidden.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])

# View the summary for each of the models]
print("\n")
print("Model-1 Summary - With one hidden layer")
model1Hidden.summary()
print("\n")
print("Model-2 Summary - With two hidden layers")
model2Hidden.summary()

# Now we will run the two models, for each of the six combinations
import datetime
accuracyList = []
start = datetime.datetime.now()
print('The training begins : ',start)
for eachCombination in grid2D:
    print("For Combination",eachCombination)
    if eachCombination[0] == 1:
        #train the model with 1 hidden layer
        model1Hidden.fit(X_train, encodedY,
                         epochs = eachCombination[1],
                         batch_size = 10)
        accuracyList.append(model1Hidden.evaluate(X_test,encodedY_test))
        
    else:
        model2Hidden.fit(X_train, encodedY, epochs = eachCombination[1],
                         batch_size = 10)
        accuracyList.append(model2Hidden.evaluate(X_test,encodedY_test))

print(list(zip(grid2D, accuracyList))) 
end = datetime.datetime.now()
print("The time taken for training the model is :",(end - start))

# From the accuracy list we see that model with 1 hidden layer and epoch = 100 gives the highest accuracy and lowest
# loss parameter

# we will then build the final model from these parameters and run it on the evaluation test

modelFinal = Sequential()
modelFinal.add(Dense(input_dim = inputDim, activation = 'sigmoid', units = int(inputDim/2),
                kernel_initializer = 'uniform', use_bias = True))
modelFinal.add(Dense(units = pageBlocks_train['CLASS'].nunique(), kernel_initializer = 'uniform', 
                use_bias = True, activation = 'sigmoid'))
# view the summary
modelFinal.summary()

# Now we will have to compile the model, we created
modelFinal.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])
# Since we are using a binary classifier, we change our cost function to binary crossentropy.
# It is also called the Log loss function

# Now we will train the model
import datetime
start = datetime.datetime.now()
print('The training begins : ',start)
modelFinal.fit(X_train, encodedY, epochs = 100, batch_size = 10)
end = datetime.datetime.now()
print("The time taken for training the model is :",(end - start))

# Now we will use the model to evaluate on test dataset
modelFinal.evaluate(X_test, encodedY_test)

# We will use this model to evaluate on the evaluation dataset

# load the eval dataset
pageBlocks_eval = pd.read_csv('//Volumes/Data/CodeMagic/Data Files/Jigsaw/page-blocks_test.csv')
# categories of the response variable
pageBlocks_eval['CLASS'].unique()

# Isolate the features and response variable
X_eval = pageBlocks_eval.drop(columns='CLASS',axis = 1)
y_eval = pageBlocks_eval['CLASS']

# Let us standardize the feature dataframe
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_eval)
X_eval = scaler.transform(X_eval)

# One hot encoding of the target variable in eval dataset
encoder = LabelEncoder()
encoder.fit(y_eval)
encodedY_eval = encoder.transform(y_eval)
encodedY_eval = np_utils.to_categorical(encodedY_eval)

# Run the predictions on the eval dataset
predictionsY_eval = modelFinal.predict_classes(X_eval)

# Evaluate the final model on the Evaluation dataset
modelFinal.evaluate(X_eval, encodedY_eval)