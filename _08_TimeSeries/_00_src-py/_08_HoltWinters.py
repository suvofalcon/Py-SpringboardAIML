"""
Holt Winters Method on airlines dataset
This method comprises of forecast equation and three smoothing equation
One for level (l_t), one for trend (b_t) and one for seasonal component (s_t) with corresponding smoothing parameters
- α, β, γ
In the previous section on Exponentially Weighted Moving Averages (EWMA) we applied Simple Exponential Smoothing using
just one smoothing factor 𝛼 α (alpha). This failed to account for other contributing factors like overall trend and
seasonality. In this section we'll look at Double and Triple Exponential Smoothing with the Holt-Winters Methods.
In Double Exponential Smoothing (aka Holt's Method) we introduce a new smoothing factor 𝛽 β (beta) that addresses trend:

Because we haven't yet considered seasonal fluctuations, the forecasting model is simply a straight sloped line
extending from the most recent data point. We'll see an example of this in upcoming lectures. With Triple Exponential
Smoothing (aka the Holt-Winters Method) we introduce a smoothing factor 𝛾 γ (gamma) that addresses seasonality:

"""

# Library imports
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset, remove missing values and validate the load
df = pd.read_csv("/Volumes/Data/CodeMagic/Data Files/Udemy/TimeSeriesData/airline_passengers.csv",
                 index_col="Month", parse_dates=True)
df.dropna(inplace=True)
print(df.head())

"""
Setting a DatetimeIndex Frequency

In holt winters method, we need to set a frequency of our date time index (whether it's daily, monthly etc.)
Since observations occur at the start of each month, we'll use MS.

A full list is as below
http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
"""
print(df.index)  # Currently the frequency is None
df.index.freq = 'MS'  # Setting it to Month Start
print(df.index)  # Check the frequency again

"""
Now we will perform simple exponential smoothing

A variation of the statsmodels Holt-Winters function provides Simple Exponential Smoothing. 
We'll show that it performs the same calculation of the weighted moving average as the pandas .ewm() method

For some reason, when optimized=False is passed into .fit(), the statsmodels SimpleExpSmoothing function shifts 
fitted values down one row. We fix this by adding .shift(-1) after fittedvalues
"""
span = 12
alpha = 2/(span + 1)

df['EWMA12'] = df['Thousands of Passengers'].ewm(span=span, adjust=False).mean()
print(df.head())

# Now we will create the model
fitted_model = SimpleExpSmoothing(df['Thousands of Passengers']).fit(smoothing_level=alpha, optimized=False)
df['SEM12'] = fitted_model.fittedvalues.shift(-1)

print("After applying EWMA and SES")
print(df.head())

# We will plot the values
df[['Thousands of Passengers', 'SEM12']].plot(title='Thousands of Passengers vs Simple Exponential Smoothing')
plt.show()

"""
Double Exponential Smoothing
Where Simple Exponential Smoothing employs just one smoothing factor 𝛼 (alpha), Double Exponential Smoothing adds a 
second smoothing factor 𝛽 (beta) that addresses trends in the data. Like the alpha factor, values for the beta factor 
fall between zero and one ( 0<𝛽≤1 )

The benefit here is that the model can anticipate future increases or decreases where the level model would only 
work from recent calculations.

We can also address different types of change (growth/decay) in the trend. If a time series displays a straight-line 
sloped trend, you would use an additive adjustment. If the time series displays an exponential (curved) trend, 
you would use a multiplicative adjustment.
As we move toward forecasting, it's worth noting that both additive and multiplicative adjustments may become 
exaggerated over time, and require damping that reduces the size of the trend over future periods until 
it reaches a flat line.
"""

fitted_model = ExponentialSmoothing(df['Thousands of Passengers'], trend='add').fit()
df['DES_add_12'] = fitted_model.fittedvalues.shift(-1)
print(df.head())

# Now we will plot the values
df[['Thousands of Passengers', 'SEM12', 'DES_add_12']].plot()
plt.show()

# We see the Double Exponential have started fitting the lines overall. In order to better visualise
# We will plot the first 24 months only
df[['Thousands of Passengers', 'SEM12', 'DES_add_12']].iloc[:24].plot().autoscale(axis='x', tight=True)
plt.show()

# Here we can see that Double Exponential Smoothing is a much better representation of the time series data.

# Now we will see, if multiplicative adjustment gives even better representation
fitted_model = ExponentialSmoothing(df['Thousands of Passengers'], trend='mul').fit()
df['DES_mul_12'] = fitted_model.fittedvalues.shift(-1)
print(df.head())
# Validate with plot
df[['Thousands of Passengers', 'SEM12', 'DES_mul_12']].iloc[:24].plot().autoscale(axis='x', tight=True)
plt.show()

# The plot shows, the multiplicative adjustment gives even better fit to the overall data

"""
Triple Exponential Smoothing
Triple Exponential Smoothing, the method most closely associated with Holt-Winters, adds support for both trends and 
seasonality in the data.
"""
fitted_model = ExponentialSmoothing(df['Thousands of Passengers'], trend='add', seasonal='add',
                                    seasonal_periods=12).fit()
df['TES_add_12'] = fitted_model.fittedvalues
print(df.head())
# Validate with plot
df[['Thousands of Passengers', 'SEM12', 'DES_add_12', 'TES_add_12']].iloc[:24].plot().autoscale(axis='x', tight=True)
plt.show()

# we have see that multiplicative model has given a better fit earlier, we would use the same
fitted_model = ExponentialSmoothing(df['Thousands of Passengers'], trend='mul', seasonal='mul',
                                    seasonal_periods=12).fit()
df['TES_mul_12'] = fitted_model.fittedvalues
print(df.head())
# Validate with plot
df[['Thousands of Passengers', 'SEM12', 'DES_mul_12', 'TES_mul_12']].iloc[:24].plot().autoscale(axis='x', tight=True)
plt.show()

# We will validate this plot on the entire dataframe
df[['Thousands of Passengers', 'SEM12', 'DES_mul_12', 'TES_mul_12']].plot().autoscale(axis='x', tight=True)
plt.show()

# We see the double exponential smoothing is much better fit than triple exponential smoothing
# Based on the plot above, you might think that Triple Exponential Smoothing does a poorer job of fitting than
# Double Exponential Smoothing. The key here is to consider what comes next - forecasting. We'll see that having the
# ability to predict fluctuating seasonal patterns greatly improves our forecast.
