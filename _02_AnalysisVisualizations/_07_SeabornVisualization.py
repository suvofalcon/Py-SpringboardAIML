"""
Demonstrates various working of Seaborn visualizations
Seaborn is a graphical library for predominantly statistical visualizations
@author: suvosmac
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn comes with some inbuilt dataset, using two such called "tips" and flights
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')

# Check the first few rows
tips.head()
flights.head()
iris.head()

# Distribution Plot showing distribution of a univariate observation
# If we want to see, how the total_bill is distributed
sns.distplot(tips['total_bill'])
plt.show()
'''
by default it comes with a histogram and density distribution
If we want to remove the distribution and show only histogram
'''
sns.distplot(tips['total_bill'], kde=False)
plt.show()

# Change the bins
sns.distplot(tips['total_bill'], kde=False, bins=30)
# Shows most of the tips bills were between 10 and 20
plt.show()

# Joint Plot - combine two distribution plots for bivariate data
sns.jointplot(x='total_bill', y='tip', data=tips)
plt.show()

''' 
Joint plot gives additional parameter called 'kind'.
By default it is scatter, if we want to show hex plots, regression plot
and density plot
'''
sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
plt.show()

sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')
plt.show()

sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde')
plt.show()

'''
We will now use pair plot to show pair-wise relationship across the 
entire data frame
'''
# pair wise relationship across all numerical columns of the data frame
sns.pairplot(tips)
plt.show()

'''
if we use the parameter hue, it will use legends for categorical column
and show it in the numerical plots..
'''
# to see all numerical relationship by the sex column
sns.pairplot(tips, hue="sex")
plt.show()
sns.pairplot(data=tips, hue='smoker')
plt.show()

'''
Now we will show Rugplot
Rugplot will show us concentration of observations across its values
it draws a dash mark for every observation
'''
sns.rugplot(tips['total_bill'])
plt.show()
# if we just want the KDE plot, we can use
sns.kdeplot(tips['total_bill'])
plt.show()

'''
Now we will explore categorical plots to plot categorical data
'''
# barplot (aggregate categorical data based on some function - default is mean)
sns.barplot(x='sex', y='total_bill', data=tips)
plt.show()

# by default the above shows average total_bill by sex, if we want to see sd instead of mean
# we use the estimator flag
sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std)
plt.show()

# if we want the estimator to specifically count the number of occurrences, we use countplot
sns.countplot(x='sex', data=tips)
plt.show()

# we will see the distribution of total bill by day and see it in a boxplot
sns.boxplot(x='day', y='total_bill', data=tips)
plt.show()
# we can add one more categorical variable in the mix to see the effect
sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker')
plt.show()
# we see that people who are smoker tend to have a larger bill on Sat and sun as opposed
# to other days

'''
violin plot - similar to box plot, showing the distribution
unlike boxplot, a violinplot, takes into account all data points and doesnt tries to figure
out the outliers
'''
sns.violinplot(x='day', y='total_bill', data=tips)
plt.show()
# we can add one more categorical variable in the mix to see the effect
sns.violinplot(x='day', y='total_bill', data=tips, hue='smoker')
plt.show()

# Stripplot - its a scatter plot, where one variable is categorical
sns.stripplot(x='day', y='total_bill', data=tips, jitter=True, hue='sex')
plt.show()

# Swarmplot is similar to strip plot,except the fact that the values are not stacked on
# top of another
sns.swarmplot(x='day', y='total_bill', data=tips, hue='sex')
plt.show()
# word of caution for swarmplot - it is computationally intensive and doesnt work well
# with large data volume

# Factor plot (renamed as catplot) is the most general method to plot the categorical variables
sns.catplot(x='day', y='total_bill', data=tips, kind='bar', hue='sex')
plt.show()

'''
Matrix plots - primarily heatmaps and clustermaps
'''
# to draw the heatmap, we need to convert our data frame into a form of matrix
# for example, if we want to plot the heatmap of the correlation between numerical variables
sns.heatmap(tips.corr(), annot=True)
plt.show()

# We can also use the pivot table to convert data into matrix form
# from the flights data, if we want to see for every month, by year, what are the number of
# passengers
fp = flights.pivot_table(index='month', columns='year', values='passengers')
# draw the heatmap with some customizations and different color palette
sns.heatmap(fp, cmap='coolwarm', linecolor='white', linewidths=1)
plt.show()

# clustermaps uses hierarchial clustering to produce a clustered version of heatmap
sns.clustermap(fp, cmap='coolwarm')
plt.show()

'''
Grids - Seaborn Grid capability to automate subplots based on feature in the data 
'''
# if we call the pairplot function on the iris dataset for various species we get
sns.pairplot(iris, hue='species')
plt.show()
# Now we can use grid function and use custom plots in every cell
g = sns.PairGrid(iris)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
plt.show()

# Using facet grid, we can create a grid based on some categorical columns
g = sns.FacetGrid(tips, col='time', row='smoker')
# now on this grid for every combination of time and smoker, we want to observe the total_bill
g.map(sns.distplot, 'total_bill')
plt.show()
# we can also observe two variables
g = sns.FacetGrid(tips, col='time', row='smoker')
g.map(plt.scatter, 'total_bill', 'tip')
plt.show()

'''
Style and Color - Now we will work with aesthetics using Seaborn 
'''
# we can set a style for our seaborn plots
plt.figure(figsize=(12, 2))  # to change the figure size
sns.set_context('poster')  # to set the context where the charts are going to be displayed
sns.set_style('whitegrid')
sns.countplot(x='sex', data=tips)
plt.show()

'''
Some Seaborn visualizations on titanic dataset
'''
titanic = sns.load_dataset('titanic')
titanic.head()

# Fair vs Age
sns.jointplot(x='fare', y='age', data=titanic)
plt.show()

# distribution of fare
sns.distplot(titanic['fare'], bins=30, kde=False, color='red', hist_kws=dict(edgecolor="k", linewidth=1))
plt.show()

sns.boxplot(x='class', y='age', data=titanic, palette='rainbow')
plt.show()

sns.swarmplot(x='class', y='age', data=titanic, palette='Set2')
plt.show()

sns.countplot(x='sex', data=titanic)
plt.show()

sns.heatmap(titanic.corr(), cmap='coolwarm')
plt.title('titanic.corr()')
plt.show()

g = sns.FacetGrid(data=titanic, col='sex')
g.map(plt.hist, 'age')
plt.show()