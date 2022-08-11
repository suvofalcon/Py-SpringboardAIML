"""
Statsmodels is a Python module that provides classes and functions for the estimation of many different statistical
models, as well as for conducting statistical tests, and statistical data exploration. An extensive list of
result statistics are available for each estimator. The results are tested against existing statistical packages
to ensure that they are correct.

The online documentation is hosted at statsmodels.org.
"""

# Library imports
from statsmodels.tsa.filters.hp_filter import hpfilter
import pandas as pd
import matplotlib.pyplot as plt

"""
The Dataset used in this exercise is US Macroeconomic Data for 1959Q1 - 2009Q3

Number of Observations - 203
Number of Variables - 14
Variable name definitions:
    year      - 1959q1 - 2009q3
    quarter   - 1-4
    realgdp   - Real gross domestic product (Bil. of chained 2005 US$,
                seasonally adjusted annual rate)
    realcons  - Real personal consumption expenditures (Bil. of chained
                2005 US$, seasonally adjusted annual rate)
    realinv   - Real gross private domestic investment (Bil. of chained
                2005 US$, seasonally adjusted annual rate)
    realgovt  - Real federal consumption expenditures & gross investment
                (Bil. of chained 2005 US$, seasonally adjusted annual rate)
    realdpi   - Real private disposable income (Bil. of chained 2005
                US$, seasonally adjusted annual rate)
    cpi       - End of the quarter consumer price index for all urban
                consumers: all items (1982-84 = 100, seasonally adjusted).
    m1        - End of the quarter M1 nominal money stock (Seasonally
                adjusted)
    tbilrate  - Quarterly monthly average of the monthly 3-month
                treasury bill: secondary market rate
    unemp     - Seasonally adjusted unemployment rate (%)
    pop       - End of the quarter total population: all ages incl. armed
                forces over seas
    infl      - Inflation rate (ln(cpi_{t}/cpi_{t-1}) * 400)
    realint   - Real interest rate (tbilrate - infl)
"""
# Let us load the dataset first
df = pd.read_csv("/Volumes/Data/CodeMagic/Data Files/Udemy/TimeSeriesData/macrodata.csv", index_col=0,
                 parse_dates=True)

# Check the data load
print(df.head())
print(df.info())

# If we plot the realgdp column
df['realgdp'].plot(figsize=(10, 4), title="Plot for realgdp")
plt.show()

# To get the trend for realgdp
# We use tuple unpacking here to get cyclical component and trend component
gdp_cycle, gdp_trend = hpfilter(df['realgdp'], lamb=1600)
# to see the type of the components - both are pandas series
print(type(gdp_cycle))
print(type(gdp_trend))

# To check the first values of gdp_trend
print(gdp_trend.head())
# to plot this data
gdp_trend.plot()
plt.show()

# Now to plot this gdp_trend against realgdp
df['trend'] = gdp_trend
df[['trend', 'realgdp']].plot(figsize=(10, 4), title='Hodrick Prescot trend vs realgdp').autoscale(
    axis='x', tight=True)
plt.show()

# On a long run they seem to overlap but there are many years where the real value seem to be above or below the
# general trend... If we need to zoom in to a particular year (everything after 2005
df[['trend', 'realgdp']]['2005-01-01':].plot(figsize=(10, 4), title='Hodrick Prescot trend vs realgdp').autoscale(
    axis='x', tight=True)
plt.show()
