from __future__ import print_function

# Spark libraries 
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

# Create a Spark Session and Initialize Application Name
spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

# Load the data as dataframe
data = spark.read.option("header", "true").option("inferSchema", "true").csv("../resources/realestate.csv")

# Now we will prepare the data for ML model to work
assembler = VectorAssembler().setInputCols(['HouseAge', 'DistanceToMRT', 'NumberConvenienceStores']).setOutputCol("features")
data = assembler.transform(data).select("PriceOfUnitArea", "features")

# Prepared Data for Machine Learning
print("Prepared Data for Machine Learning")
data.show(10)

# Performing a training and test split
train_data, test_data = data.randomSplit([0.7, 0.3])

# We will now initialize the decision tree model
dtModel = DecisionTreeRegressor().setFeaturesCol("features").setLabelCol("PriceOfUnitArea")

# Train the model using our training data
dtModel = dtModel.fit(train_data)

# Using the model, we will predict from our test data
# Now we are going to evaluate this model on test data
unlabeled_data = test_data.select('features')
predictions = dtModel.transform(unlabeled_data)
print("\n")
print("Showing Predictions")
predictions.show(10)

# Now we will display the actual and predictions side by side
compare = test_data.join(predictions, test_data.features == predictions.features)
compare.show(10)

# Close the Spark Session
spark.stop()