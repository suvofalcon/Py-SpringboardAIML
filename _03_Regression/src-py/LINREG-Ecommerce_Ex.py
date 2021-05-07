# -*- coding: utf-8 -*-
"""
Linear Regression With Python (Udemy Exercise) - LINREG-Ecommerce_Ex.py

You just got some contract work with an ECommerce company based in New York City that sells clothing online but
they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings 
with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
The company is trying to decide whether to focus their efforts on their mobile app experience or their website. 
They've hired you on contract to help them figure it out! Let's get started!

@author: suvosmac
"""

# library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# read the dataset
customers = pd.read_csv('//Volumes/Data/CodeMagic/Data Files/Udemy/Ecommerce Customers')

# check the head 
customers.head()
# check the structure
customers.info()
# check the summary statistics
customers.describe()

# Exploratory Data Analysis
# Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns.
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
plt.show()
# The above shows more time on website, more money spent (very little positive correlation)
# using Time on App column
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
plt.show()
# Strong positive correlation ...more time on App, more Amount spent

# Using jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.
sns.jointplot(x='Time on App', y='Length of Membership', kind='hex', data=customers)
plt.show()

# Let's explore these types of relationships across the entire data set. Use pairplot
sns.pairplot(customers)
plt.show()
# Lets plat a heatmap to understand correlation across the data frame
sns.heatmap(customers.corr(), annot=True)
plt.show()

# linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
plt.show()

# split the data into training and testing sets. target variable is Yearly Amount Spent

target = customers['Yearly Amount Spent']
features = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

# Use model_selection.train_test_split from sklearn to split the data into training and testing sets. 
# Set test_size=0.3 and random_state=101
# from sklearn.model_selection import train_test_split
# Now we will split it using tuple unpacking
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3,
                                                                            random_state=101)

# Training the Model - Now its time to train our model on our training data!

# creating an instance of linear regression
lm = LinearRegression()

# fit the model on the training data
lm.fit(features_train, target_train)

# See the coefficients of the model
lm.coef_
# We will create a data frame to relate these coefficients with respective columns
coefDf = pd.DataFrame(lm.coef_, features_train.columns, columns=['Coeff'])
coefDf.head()

# Predicting Test Data - Now that we have fit our model, 
# let's evaluate its performance by predicting off the test values!
predictions = lm.predict(features_test)

# we will plot the actuals vs predicted
# create a diagnostic dataframe
diagDf = pd.DataFrame({'Actuals': target_test, 'Predictions': predictions})
sns.lineplot(data=diagDf)
plt.xlabel("Actuals")
plt.ylabel("Predicted")
plt.title("Actuals vs Predicted")
plt.show()

# Evaluating the Model
# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.

print('MAE:', metrics.mean_absolute_error(target_test, predictions))
print('MSE:', metrics.mean_squared_error(target_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(target_test, predictions)))

# Explained variance score
metrics.explained_variance_score(target_test, predictions)  # 93.1 variations in dv is explained

# Residuals
# Plot a histogram of the residuals and make sure it looks normally distributed. 
sns.distplot((target_test - predictions))
plt.show()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Another approach to run the linear regression
'''
import statsmodels.api as sm

idv = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
dv = customers['Yearly Amount Spent']
idv_train, idv_test, dv_train, dv_test = train_test_split(idv, dv, test_size=0.3, random_state=101)
model = sm.OLS(dv_train, idv_train).fit()
model.summary()

# to check variable inflation factors
# For each Xi, calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(idv_train.values, i) for i in range(idv_train.shape[1])]
vif['features'] = idv_train.columns
vif

# Now running the predictions
predictions = model.predict(idv_test)
plt.scatter(dv_test, predictions)
plt.show()

residuals = dv_test - predictions

# we will now, test for heteroskedasticity
plt.scatter(predictions, residuals)
plt.xlabel("Predictions")
plt.ylabel("Residual Errors")
plt.title("Predictions vs Residuals")
plt.show()
