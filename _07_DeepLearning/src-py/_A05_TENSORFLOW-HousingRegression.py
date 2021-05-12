"""
We will demonstrate tensor flow for a Regression

California Housing Data
This data set contains information about all the block groups in California from the 1990 Census. In this sample a
block group on average includes 1425.5 individuals living in a geographically compact area.
The task is to aproximate the median house value of each block from the values of the rest of the variables.

The Features:
-housingMedianAge: continuous.
-totalRooms: continuous.
-totalBedrooms: continuous.
-population: continuous.
-households: continuous.
-medianIncome: continuous.
-medianHouseValue: continuous.

@author: suvosmac

- We will model this using
- Tensorflow (DNNRegressor)
- Tensorflow (LinearRegressor)
- Without using Tensorflow - simple OLS regression

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import  tensorflow as tf

# Lets load the data
housing = pd.read_csv('/Users/suvosmac/Documents/CodeMagic/DataFiles/Udemy/cal_housing_clean.csv')

# Check the data load
housing.head()
housing .info()

# See the summary statistics
housing.describe()

# See the summary statistics specifically for the target variable
housing['medianHouseValue'].describe()

# seggregate the dependent and independent variable(s) dataframes

x_data = housing.drop(['medianHouseValue'],axis=1)
y_val = housing['medianHouseValue']

# Lets do some visualizations on the data
# study the distribution of the target variable - most of the data points are at a lower range
sns.distplot(housing['medianHouseValue'],kde=False,bins=10)
# self relationships on the entire datasets
sns.pairplot(housing)
# heatmap on correlations
sns.heatmap(housing.corr(),annot=True)

'''
Perform Train, test split
Scale feature data
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data,y_val,test_size=0.3,random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(data=scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
X_test = pd.DataFrame(data=scaler.transform(X_test),columns=X_test.columns,index=X_test.index)

'''
Create feature columns
Create the input function for the model
'''
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

feat_cols = [age,rooms,bedrooms,pop,households,income]

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,
                                                 shuffle=True)

'''
We will build the model using DNNRegressor - Densely Connected Neural network

Build the neural network model
Predict from the model
Evaluate the model
'''
model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=feat_cols)
# train the model for 25000 steps
model.train(input_fn=input_func,steps=25000)

# prediction input function
predict_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)
predictions = list(model.predict(predict_input_func))

print(predictions)

final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])

# Lets plot the actuals vs predicted
plt.scatter(y_test,final_preds,color=['red','blue'])
plt.xlabel("Actuals")
plt.ylabel("Predicted")
plt.title("Actuals vs Predicted")

# Evaluating the Model
# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, final_preds))
print('MSE:', metrics.mean_squared_error(y_test, final_preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, final_preds)))

# Explained variance score
metrics.explained_variance_score(y_test,final_preds)

'''
We will build the model using LinearRegressor 

Build the neural network model
Predict from the model
Evaluate the model
'''

estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# we will define the input functions
train_input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,
                                                       shuffle=True)
test_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)

estimator.train(input_fn=train_input_func,steps=25000)

# predictions
predictions = list(estimator.predict(test_input_func))

print(predictions)

final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])

# Lets plot the actuals vs predicted
plt.scatter(y_test,final_preds,color=['red','blue'])
plt.xlabel("Actuals")
plt.ylabel("Predicted")
plt.title("Actuals vs Predicted")

# Evaluating the Model
# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, final_preds))
print('MSE:', metrics.mean_squared_error(y_test, final_preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, final_preds)))

# Explained variance score
metrics.explained_variance_score(y_test,final_preds)

'''
We will build the model using Simple Linear Regression and 
NOT use any neural network
'''

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data,y_val,test_size=0.3,random_state=101)

# build the OLS regression model
model = sm.OLS(y_train,X_train).fit()
# Check the summary
model.summary()

# Now running the predictions
predictions = model.predict(X_test)
plt.scatter(y_test,predictions,color=['red','blue'])
residuals = y_test-predictions

# to check variable inflation factors
# For each Xi, calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['features'] = X_train.columns
vif


# we will now, test for heteroskedasticity
plt.scatter(predictions,residuals)
plt.xlabel("Predictions")
plt.ylabel("Residual Errors")
plt.title("Predictions vs Residuals")

# Evaluating the Model
# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Explained variance score
metrics.explained_variance_score(y_test,predictions)