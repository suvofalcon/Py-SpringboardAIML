"""
Data Visualization with Pandas

@author: suvosmac
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lets load the dataframe

df1 = pd.read_csv("Data/df1.csv", index_col=0)
df1.head()

'''
Plot histograms
'''
df1['A'].plot.hist()
plt.show()

df1['A'].plot.hist(edgecolor='k')  # adding edges to the bar
plt.show()

# plotting taking the whole area
df1['A'].plot.hist(edgecolor='k').autoscale(enable=True, axis='both',
                                            tight=True)
plt.show()

# Adding bins
df1['A'].plot.hist(bins=20, edgecolor='k')
plt.show()

df1['A'].plot.hist(bins=10, edgecolor='k', grid=True)
plt.show()

'''
Plot barplots
'''
df2 = pd.read_csv("data/df2.csv")
df2.head()

df2.plot.bar()
plt.show()

df2.plot.bar(stacked=True)
plt.show()

df2.plot.barh()
plt.show()

'''
Line Plots
'''
df2.plot.line(y='a', figsize=(10, 4))
plt.show()

df2.plot.line(y=['a', 'b', 'c'], figsize=(10, 4), lw=4)
plt.show()

'''
Area Plots
'''
df2.plot.area(alpha=0.4)  # alpha is transparency
plt.show()

'''
scatter plots
'''
df1.plot.scatter(x='A', y='B')
plt.show()

df1.plot.scatter(x='A', y='B', cmap='coolwarm')
plt.show()

# If we want to do size instead of colorby
df1.plot.scatter(x='A', y='B', s=df1['C']*50, alpha=0.35, cmap='coolwarm')
plt.show()

'''
Box Plots and KDE Plots (kernel density estimation)
'''
df2.plot.box()
plt.show()

df2.plot.kde()
plt.show()

'''
Hexagonal Bin Plots
'''
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df.plot.scatter(x='a', y='b')
plt.show()

df.plot.hexbin(x='a', y='b', gridsize=25, cmap='Oranges')
plt.show()

'''
Customizing plots created by Pandas
'''
df2['c'].plot.line(figsize=(20, 3), ls=':', c='red', lw=5)
plt.show()

title = "MY PLOT TITLE"
xlabel = 'My X Data'
ylabel = 'My Y Data'

ax = df2['c'].plot.line(figsize=(10, 3), ls=':', c='red', lw=5, title=title)
ax.set(xlabel=xlabel, ylabel=ylabel)
plt.show()

ax = df2.plot()
ax.legend(loc=0, bbox_to_anchor=(1.0, 1.0))  # moving the legend outside the plot
plt.show()