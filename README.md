# Py-SpringboardML
This will have code demonstration for all aspects of machine learning, deep learning, computer vision, automation, feature engineering implementations using Python

## Crash Course

This will have illustrative examples around Python Numpy and Pandas packages - Crash course before we delve into ML

* Numpy basics
* Numpy Arrays
* Nummpy Indexing
* Numpy Operations
* Pandas basics
* Pandas Series
* Pandas Dataframes
* Pandas Operations (missing daya, groupby, merging, joining and concatenating)
* Pandas Visualizations

## Language Basics

This will have some fundamental concepts of Python as a language explained

* Language Basics
* Decisions and Loops in Python
* User defined functions in Python
* Fundamentals of File Operations in Python

Source Code showing various aspects of the language Code snippets for

* Working with Numpy package
* Basics of using Pandas Package
* How to use Pandas Package for Visualization
* Demonstrating Pandas Packages in two datasets
* SF Salaries Dataset
* ECommerce Purchases Dataset
* Demonstrating some basic concepts of matplotlib
* Demonstrating some basic concepts for seaborn
* Exploratory Data analysis on 911 calls databases
* Analysis of a finance stock data (this is optional) - The data source is not working

## AnalysisVisualizations

This will have Python Source files demonstrating fundamentals of Python Language Features Usage of some additional packages used in Data analysis and Visualizations in Python

* Pandas Visualizations
* Matplotlib Visualizations
* Seaborn Visualizations

## Regression

Machine Learning Algorithms for Regression type Problems

* LINREG - Linear Regression  
* LOGREG - Logistic Regression
* POLYREG - Polynomial Regression
* SVR - Support Vector Regressor
* DTREE - Decision Tree Regressor
* RANDFOREST - Random Forest Regression

## Classification Algorithms

Machine learning algorithms for classification type problems

* DTREERNDFOREST - Decision Trees and Random Forest
* KMEANS - K-Means algorithm for Classification
* KNN - K Nearest neighbour algorithm
* NLP - Natural Language Processing
* NBAYES - Naive Bayes Classifier

* PCA - Principal Component Analysis
    * Implementation of Principal Component Analysis on breast cancer dataset, thereafter using those principal components in Logistic regression performing breast cancer classification

* SVM - Support Vector Machines for Classification

## Recommenders

Various recommendation algorithms

* A simple movie recommendations based on collaborative filtering

## Big Data

Various implementations within Big Data technologies

## Deeep Learning 

This has various implementations of neural networks

* Artificial Neural Network implementation to predict customer churn
* ANN Implementation to predict credit card default
* Convolutional Neural Network (CNN) implementation to classify Cats-Dogs images 
* Recurrent Neural Network (RNN) with LSTM layers implementation to predict Google Stock prices
* Implementing Self Organizing Maps (Unsupervised Algorithm) to implement Fraud Detection
* Implementing an hybrid model to move from Unsupervised to Supervised deep learning model. Self Organizing Maps and Aritificial Neural networks combined.
* Implementing Page Blocks Classification - multi class classification using a Multi layer NN in TF 2.0
* Implementation of Recommender Systems using Boltzmann Machines

### Source Folder

The source folder has some miscelleneous test files

* If using <em>GPU for NN training</em>, how to make GPU memory growth automatic and gradual - using tensorflow as backend for Keras
* Work in progress implementation of how achieve parallelism by creating virtual gpus on a physical gpu and run multiple tensorflow jobs on each virtual gpu

### NLP - Natural Language Processing

An exhaustive implementation of various NLP techniques using ML and Deep Learning

* Python Text Basics
	* Working With Text Files
	* Working PDF Files
	* Regular Expressions
* NLP Basics
	* Spacy Library Basics
	* Tokenization
	* Stemming
	* Lemmatization
	* Usage of Stop Words
	* Vocabulary and Matching
* Parts of Speech and Named Entity Recognition
	* Parts of Speech Basics
	* Visualization of Parts of Speech
	* Named Entity Recognition
	* Visualizing Named Entity Recognition
	* Sentence Segmentation
* Text Classification
	* A primer of Scikit Learn Library - How to use Scikit learn and use the models within
	* Feature Extraction from Text Data
	* Text Classification on Sample movie reviews data
* Semantics and Sentiment Analysis
	* Semantics and Word Vectors (Uses Spacy and language model)
	* Sentiment Analysis using NLTK and VADER on sample amazon reviews dataset
	* Sentiment Analysis using NLTK and VADER on sample moview reviews dataset
* Topic Modelling
	* Latent Dirichilet Allocation for Topic Modelling
	* Non Negative Matrix Factorization for Topic Modelling
	* Using Non Negative Matrix Factorization for Topic Modelling on a sample Quora dataset


## TimeSeries Forecasting and Analysis

* DateTime Objects in Numpy and Pandas
* Resampling, Shifting, Rolling and Expanding of of Time Series Data
* Time Series Visualizations
* Introduction to Statsmodels
* ETS Decomposition
* Exponentially Weighted Moving averages (EWMA)
* Holt-Winters method - Double and Triple Exponential smoothing
* A sample exercise on the <em>statsmodels.ts</em>
* Introduction to Forecasting
* Stationarity in Time Series
* Auto Correlation and Partial Auto Correlation Functions (plots)
* Auto Regression Models (AR Models)
* Some Descriptive Statistics and Tests
* Chosing ARIMA Orders for modelling
* ARMA and ARIMA Models
* Seasonal ARIMA Models (SARIMA)
* Seasonal ARIMA with Exogenous Variables (SARIMAX	)
* Vector Autoregressive Models (VAR)
* Vector Autoregressive Moving Average Models (VARMA)
* Tensorflow Keras basics
* Using Recurrent Neural Networks (RNN) - LSTM Networks to forecast timeseries data (Tensorflow backend was used)
* Introduction to Facebook Prophet Library
* Forecasting and Diagnosting a timeseries using Facebook Prophet Library

## Tensor Flow 2.0

Detailed Samples on how to use Tensor Flow 2.0 (Latest release of TensorFlow. Most of these examples uses Tensorflow gpu version of the library. These are tested on NVIDIA RTX 2070EX Super)

* Tensor Flow Basics - How to use Tensorflow 2.0 constants, variables and strings and tensor operations
* A multi layer NN Model for Classification - Kaggle Fashion MNIST dataset
* A basic single layer NN model for Regression -  Convert Celsius to Fahrenheit
* A basic single layer NN moodel for Regression - Predicting Sales of Ice Creams using Outside Temperature
* A multi Layer NN model for Regression - Predicting Bike Rental Usage on a real data
* A multi layer NN model for Regression - Predicing housing prices of King County, Washington (Kaggle)
* A multi layer NN model for Classification - Text Classification using Amazon Alexa reviews (Kaggle)
* A multi layer NN model for Classification - Diabetes Outcome Classification
* A multi layer NN model for multi class Classification - Page Blocks Classification
* Classification on Imbalanced Data <font color='blue'>This is still a work in progress</font>
* Building a Convolutional Neural Network on Fashion MNIST dataset
* Building a Convolutional Neural Network on Traffic Sign Classification (LeNet - Yann LaCunn's original presentation)
* Building a Convolutional Neural Network on CIFAR10 dataset
* Building a RNN LSTM Network to predict stock prices
* Building a Q&A Chatbot with Tensorflow by implementing LSTM (This is a WIP file)

## PySpark

This has many implementations of how to use PySpark library on Apache Spark. The Spark version used is 2.4.5 on Hadoop 2.7.x

* Demonstration of PySpark Library basics
* Linear Regression Examples using PySpark
* Logistic Regression Examples using PySpark
* Tree Methods (Decision Tree, Random Forests, Gradient Boosted Trees) using PySpark
* KMeans Clustering using PySpark
* Recommender system by Collaborative Filtering using PySpark ALS method
* Natural Language Processing using PySpark
* Spark Streaming using PySpark

## Feature Engineering

This covers the salient features of feature engineering for Machine Learning

## Python Automation

This covers basic usage of Python on automation tasks