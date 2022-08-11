"""
A Sample exercise on Statsmodels

For this set of exercises we're using data from the Federal Reserve Economic Database (FRED) concerning the
Industrial Production Index for Electricity and Gas Utilities from January 1970 to December 1989.

Data source: https://fred.stlouisfed.org/series/IPG2211A2N
"""
# library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Global Constants
span = 12
alpha = 2/(span + 1)

# Load and validate the data
df = pd.read_csv("/Volumes/Data/CodeMagic/Data Files/Udemy/TimeSeriesData/EnergyProduction.csv",
                 index_col='DATE', parse_dates=True)
print(df.head())

# set the frequency to our date time index
df.index.freq = 'MS'
print(df.index)

# We will now plot the dataset
df.plot(title='Energy Index').autoscale(axis='x', tight=True)
plt.show()

# Add a column that shows a 12-month Simple Moving Average (SMA).
# Plot the result.
df['SMA12'] = df['EnergyIndex'].rolling(window=12).mean()
df.plot(title='Energy Index vs SMA').autoscale(axis='x', tight=True)
plt.show()

# Add a column that shows an Exponentially Weighted Moving Average (EWMA) with a span of 12 using
# the statsmodels SimpleExpSmoothing function. Plot the result.
fitted_model = SimpleExpSmoothing(df['EnergyIndex']).fit(smoothing_level=alpha, optimized=False)
df['SES12'] = fitted_model.fittedvalues.shift(-1)
df.plot(title='EnergyIndex vs SMA vs EWMA').autoscale(axis='x', tight=True)
plt.show()

# Add a column to the DataFrame that shows a Holt-Winters fitted model using Triple Exponential Smoothing with
# multiplicative models. Plot the result.
fitted_model = ExponentialSmoothing(df['EnergyIndex'], trend='mul', seasonal='mul', seasonal_periods=12).fit()
df['TES_MUL_12'] = fitted_model.fittedvalues
print(df.head())
df.plot(figsize=(12, 6), title='EnergyIndex vs SMA vs EWMA vs TES').autoscale(axis='x', tight=True)
plt.show()

#  OPTIONAL: Plot the same as above, but for only the first two years.
df[:24].plot(figsize=(12, 6), title='EnergyIndex vs SMA vs EWMA vs TES').autoscale(axis='x', tight=True)
plt.show()