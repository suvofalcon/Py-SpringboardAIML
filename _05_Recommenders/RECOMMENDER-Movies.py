# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:32:08 2018

Demo a simple recommendation system in Python using item similarity

The two most common types of recommender systems are Content-Based and Collaborative Filtering (CF)

- Collaborative Filtering produces recommendations based on the knowledge of users' attitude to items, that is, it uses
the "wisdom of the crowd" to recommend items

- Content-based recommender systems focus on the attributes owf the items and give you recommendations based on the
similarity between them

In general, Collaborative Filtering(CF) is more commonly used than content-based systems because it usually gives
better results and is relatively easy to understand(from an overall implementation perspective)
The algorithm has the ability to do a feature learning of its own, which means that it can start to learn for itself, 
what features to use
CF can be further divided into memory based collaborative filtering and model based collaborative filtering
Model Based CF - Uses singular value decomposition
Memory Based CF - Cosine Similarity

Content based recommender system for a dataset of movies

@author: 132362
"""
# import libraries
import pandas as pd
import seaborn as sns

# create a list variable
columns_names = ['user_id', 'item_id', 'rating', 'timestamp']

# load the data
df = pd.read_csv("D:\\CodeMagic\\Data\\u.data", sep='\t', names=columns_names)
# Check the initial tows
df.head()

# Now we will grab the movie title
movie_titles = pd.read_csv("D:\\CodeMagic\\Data\\Movie_Id_Titles")
# Check the initial rows
movie_titles.head()

# Now we will merge these two on item_id
df = pd.merge(df, movie_titles, on='item_id')
# Check the rows
df.head()

# By Movies, I would want to find out the average rating
df.groupby('title')['rating'].mean()
# To order them by seeing the highest ratings
df.groupby('title')['rating'].mean().sort_values(ascending=False)
# since the above was just a groupby by ratings and average, it might so happen that this movie
# was seen by very few number of people but all high ratings means average rating also high

# Movies with most ratings - maximum number of people have seen this and rated
df.groupby('title')['rating'].count().sort_values(ascending=False)

# Lets create a dataframe called ratings
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
# Check the head of the dataframe
ratings.head()
# add the number of ratings columns
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()

# Visually explore this data frame
ratings['num of ratings'].hist(bins=70)
# We see most of our number of ratings is low

# now see the ratings column
ratings['rating'].hist(bins=70)
# Most of the ratings falls within 3 to 4

# see the relationship between actual number of ratings and average rating
sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)
# The above shows as the num of ratings increases, we are more likely to get a higher rating for a movie

'''
Build the recommendation system for the movie titles
'''

# We will construct a matrix which will have user id on one axis and movie title on another axis
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
# Check the head of it
moviemat.head()
# There are lot of missing values, because most of the people, have not see most of the movies

# Lets do a sort by the most watched movies - top 10
ratings.sort_values('num of ratings', ascending=False).head(10)

# Lets find out the ratings for two movies from the matrix
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

# Check the ratings for one move
starwars_user_ratings.head()

# Lets find the correlation between starwars user ratings and user ratings for other movie titles
# correlation with every other movie with that specific user behaviour for starwars
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

# We will use a dataframe
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
# Drop the NA value
corr_starwars.dropna(inplace=True)
# Check the data frame
corr_starwars.head()  # How correlated these movies user ratings were with respect to star wars user rating

# Now we will join the number of ratings columns
corr_starwars = corr_starwars.join(ratings['num of ratings'])
# Now we will filter out any movies which does not have atleast 100 ratings
corr_starwars[corr_starwars['num of ratings'] > 100].sort_values('Correlation', ascending=False).head(10)

# do the same with liar liar
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
# Drop the NA value
corr_liarliar.dropna(inplace=True)
# Now we will join the number of ratings columns
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
# Now we will filter out any movies which does not have atleast 100 ratings
corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values('Correlation', ascending=False).head(10)

# We can play around with the num of ratings filter and see how the correlation values differ and come to a 
# recommendation

