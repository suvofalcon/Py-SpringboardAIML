'''
We will build a neural network using the iris dataset

@author : Subhankar
'''
import pandas as pd
import seaborn as sns

# Load the iris dataset
iris = pd.read_csv("//Volumes/Data/CodeMagic/Data Files/Jigsaw/iris.csv")

# Check the dimensions of the dataset, info and initial rows
print (iris.shape)
# check summary statistics
iris.describe()
# quick info on the dataset
iris.info()
# check initial 5 rows
iris.head()

'''
Some exploratory analysis
'''
# Study pairplot to get relationship across variables except the Species and Response columns
sns.pairplot(iris,hue='Species')
'''
Some observations come out quite strongly
- Sepal Length seems to be correlated with Petal length
- Sepal Width seems to be correlated with Petal Width
These insights can be used to reduce the number of variables to consider

- There is strong linear decision boundary between Setosa and other species, 
but there is no linear decision boundary between Versicolor and Virginica
'''

'''
Before we start modelling
We will encode the species names to numerical columns
'''
# find out the unique labels of the response column
iris['Species'].unique()
# we create a dictionary
encoder = {'setosa' : 0,'versicolor' : 1,'virginica' : 2}
# will add the numerical response column
iris['response'] = [encoder[eachVal] for eachVal in iris.Species]

'''
We will fit a basic neural network.
First will isolate the feature variables
Second - Will perform one hot encoding on the response variable. When working with a
a multi class classification problem, keras expects to perform one hot encoding on the
response variable. If the response is a categorical variable, keras expects dummy variables 
to be introduced for every label of the class

Important point to remember, if we have features which are not numerical,(characters or strings)
representing some groupings, one hot encoding has to be performed on those as well
'''
features = iris[iris.columns.values[:4]].values
# Check the first 4 values of the array
features[:4]

# One hot encoding
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(iris['Species'])
encoded_Y = encoder.transform(iris['Species'])
responseEnc = np_utils.to_categorical(encoded_Y)
responseEnc[:4]

'''
Now we start training the neural network

We  will build a neural network with one layer which will have three neurons.
Each of this neuron will take four features and passes the weighted sum through a 
transformation function and will produce an output

Each neuron corresponds to each class in the response variable

In keras terminology, this type of structure is called sequential model
'''
from keras.models import Sequential
from keras.layers import Dense, Activation

# Initialize the model. Neural networks like this are sequential model in keras terminology
model = Sequential()
'''
We will then add the first layer and the only layer in our neural network model
This layer will have three neurons and each neuron will have four features as its input.
This is done using the add method and Dense function specifies all these to be added
to a neural network layer

init specifies the initial values of weight to be used. Gradient descent uses some initial
values of the weights to kickstart the algorithm and one of the method used in keras is called
'uniform' which sets the initial values from a randomly drawn samples from a uniform distribution

bias specifies whether a bias node is needed in the model or not
activation function to be used
'''
# add input layer
model.add(Dense(input_dim = 4, activation = 'sigmoid', units = 3, kernel_initializer = 'uniform',
                use_bias = True))
#model.add(Dense(output_dim = 3, input_dim = 4, init = 'uniform', bias = True,
#               activation = 'sigmoid'))

# Check the model summary
print(model.summary())
'''
The next step is to compile the model, which configures the learning process
The first argument is the loss, which is the cost function that we want to optimise

The second is the name of the optimiser function. The one which we use is called 'adam' 
-adaptive moment estimation
Finally the metrics argument specifies how the model should be evaluated. Here to keep this
simple, we mention 'accuracy'
'''
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',
              metrics = ['accuracy'])

'''
Once the model has been configured to run, its time to train the data
The fit method for model starts the training method

The first two arguments specify the features and response we want to use for training the
model. The third argument is the number of times the algorithm iterates on the data.
Higher the number, more is the time for training
The last one is the number of samples used to update the weights for the next iterations
'''
model.fit(features, responseEnc, nb_epoch = 500, batch_size = 10)

'''
Once the training is done, we will use the model for predictions
'''
model.predict_classes(features)

'''
Find the model evaluation score
'''
scores = model.evaluate(features,responseEnc, verbose=0)
print("Accuracy in prediction %0.2f" %(scores[1]*100))
