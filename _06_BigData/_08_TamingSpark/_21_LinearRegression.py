from __future__ import print_function

# Spark Libraries import

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# Intialize the Spark Session and set application name
spark = SparkSession.builder.appName('LinearRegression').getOrCreate()

# We define a custom Schema
# We define a schema which has to be a struct type
schema = StructType([StructField("label", DoubleType(), True),
                    StructField("feature", DoubleType(), True)])

# Read the dataset

data = spark.read.schema(schema=schema).csv("../resources/regression.txt")
data.show(10)

# Now we will need to prepare the data in the format Spark ML expects before the model is defined and trained
assembler = VectorAssembler(inputCols=['feature'], outputCol="features")
output = assembler.transform(data)

print("Check on the Schema after transformation")
output.printSchema()

# Now select the final_data
final_data = output.select("label", "features")

# We will split the data into training and test
train_data, test_data = final_data.randomSplit([0.7, 0.3])

# # Now we will create our linear regression model
lrmodel = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)

# # We will train the model using our training data
lrmodel = lrmodel.fit(train_data)

# # Get the coefficients and Intercepts
print("\n")
print(f"coefficients : {lrmodel.coefficients}")
print(f"Intercept : {lrmodel.intercept}")

# Using the model, we will predict from our test data
# Now we are going to evaluate this model on test data
test_results = lrmodel.evaluate(test_data)

# Get the residuals, RMSE and Adjusted R2
test_results.residuals.show(10)
print("\n")
print(f"RMSE : {test_results.rootMeanSquaredError}")
print(f"MSE : {test_results.meanSquaredError}")
print(f"R2 : {test_results.r2}")
print(f"Adjusted R2 : {test_results.r2adj}")

# Now if we have to run these as predictions on an unseen data

unlabeled_data = test_data.select('features')
predictions = lrmodel.transform(unlabeled_data)
print("\n")
print("Showing Predictions")
predictions.show()

# Now we will display the actual and predictions side by side
compare = test_data.join(predictions, test_data.features == predictions.features)
compare.show(10)

# Close the spark session
spark.stop()


