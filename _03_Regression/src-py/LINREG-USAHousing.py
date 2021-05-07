# -*- coding: utf-8 -*-
"""
Linear Regression With Python

A real estate agent and wants some help predicting housing prices for regions in the USA. 
It would be great if you could somehow create a model for her that allows her to put in a few 
features of a house and returns back an estimate of what the house would sell for.

The data contains the following columns:
'Avg. Area Income': Avg. Income of residents of the city house is located in.
'Avg. Area House Age': Avg Age of Houses in same city
'Avg. Area Number of Rooms': Avg Number of Rooms for Houses in same city
'Avg. Area Number of Bedrooms': Avg Number of Bedrooms for Houses in same city
'Area Population': Population of city house is located in
'Price': Price that the house sold at
'Address': Address for the house

"""

# Lets import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lets import the data
USAhousing = pd.read_csv('/Volumes/Data/CodeMagic/Data Files/Udemy/USA_Housing.csv')
# view the initial rows
USAhousing.head()
# lets view the structure of the data
USAhousing.info()
# quickly check for any missing values in the dataframe
print(USAhousing.isnull().sum())

# lets examine the summary statistics of the data frame
USAhousing.describe()
# to quickly view the column names of a data frame
USAhousing.columns

# We will now do some visualizations to check the data. We can run this visualization if the
# data frame is not very large
sns.pairplot(USAhousing)
plt.show()

# lets check the distribution of our target variable
sns.distplot(USAhousing['Price'])  # looks to be the average price is b/w 1-1.5 million
plt.show()

# We can observe the entire correlation in the data frame and draw a heatmap for the samer
sns.heatmap(USAhousing.corr(), annot=True)
plt.show()

# Let us study the relationship between income and price
sns.jointplot(x='Avg. Area Income', y='Price', data=USAhousing)
plt.show()

# Now we will attempt to model a linear regression

# First we will separate out the feature variables and the target variable
# We will not use the address column in the Linear regression model
features = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                       'Avg. Area Number of Bedrooms', 'Area Population']]
target = USAhousing['Price']

# Now we will split the data into training and test using scikit learn
from sklearn.model_selection import train_test_split

# Now we will split it using tuple unpacking
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.4,
                                                                            random_state=101)

# Lets import the model from the linear regression family
from sklearn.linear_model import LinearRegression
# create and instantiate a linear regression object
lm = LinearRegression()

# Now lets attempt to fit the model on training data
lm.fit(features_train, target_train)

# Now lets evaluate the model by checking its coefficients
# intercept
lm.intercept_
# Coefficients
lm.coef_

# We will create a data frame to relate these coefficients with respective columns
coefDf = pd.DataFrame(lm.coef_, features_train.columns, columns=['Coeff'])
coefDf.head()

# Predictions from our model
predictions = lm.predict(features_test)  # in predict method, we pass just the test data

# Now we will visually analyse the actual vs predictions
'''
The above shows almost a straight line with points from both actuals and predictions overlapping each other.
This signifies the model has done quite a good job in predictions
'''
plt.scatter(target_test, predictions)
plt.show()

# histogram of the distribution of our residuals (errors)
'''
We see the residuals are normally distributed with mean at 0. which means the model is a good choice
'''
sns.distplot((target_test - predictions))
plt.show()

'''
Regression Evaluation Metrics¶
Here are three common evaluation metrics for regression problems:
    
Mean Absolute Error (MAE) is the mean of the absolute value of the errors:
 
Mean Squared Error (MSE) is the mean of the squared errors:
 
Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:
 
Comparing these metrics:
MAE is the easiest to understand, because it's the average error.
MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
All of these are loss functions, because we want to minimize them.
'''

# to calculate all of these
from sklearn import metrics

print('MAE  ', metrics.mean_absolute_error(target_test, predictions))  # MAE
print('MSE  ', metrics.mean_squared_error(target_test, predictions))  # MSE
print('RMSE  ', np.sqrt(metrics.mean_squared_error(target_test, predictions)))

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Another approach to run the linear regression
'''

# We see Avg. Area Number of Bedrooms and Avg. Area Number of Rooms is not statistically significant,
# so dropping those variables

idv = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Area Population']]
dv = USAhousing['Price']

# Now we will split it using tuple unpacking
idv_train, idv_test, dv_train, dv_test = train_test_split(idv, dv, test_size=0.4, random_state=101)
import statsmodels.api as sm
model = sm.OLS(dv_train, idv_train).fit()
model.summary()

# We observe this model has lot of multicollinearity and hence dropping two idvs and re-run

idv = USAhousing[['Avg. Area Income']]
dv = USAhousing['Price']

# Now we will split it using tuple unpacking
idv_train, idv_test, dv_train, dv_test = train_test_split(idv, dv, test_size=0.4, random_state=101)
model = sm.OLS(dv_train, idv_train).fit()
model.summary()

''' Not needed any more as idv is just 1
# to check variable inflation factors
# For each Xi, calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(idv_train.values, i) for i in range(idv_train.shape[1])]
vif['features'] = idv_train.columns
vif
'''

# Now running the predictions
predictions = model.predict(idv_test)
plt.scatter(dv_test, predictions)
plt.show()

diagDf = pd.DataFrame({'Test': dv_test, 'Pred': predictions})
sns.lineplot(data=diagDf)
plt.show()

# Plotting of residuals
residuals = dv_test-predictions
sns.distplot(residuals)
plt.show()

# we will now, test for heteroskedasticity
plt.scatter(predictions, residuals)
plt.xlabel("Predictions")
plt.ylabel("Residual Errors")
plt.title("Predictions vs Residuals")
plt.show()