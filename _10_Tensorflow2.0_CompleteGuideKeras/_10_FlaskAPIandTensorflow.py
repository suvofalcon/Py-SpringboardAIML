"""
# First we save some file from our mnist dataset to uploads folder
from tensorflow.keras.datasets import fashion_mnist
from matplotlib.pyplot import imsave

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_test.shape)
for index in range(5):
    imsave(fname="resources/uploads/{}.png".format(index), arr=X_test[index])
"""


# Import all project dependencies


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, jsonify

print(f"Tensorflow Version getting used - {tf.__version__}")

# Now we will load the pre-trained model
with open('resources/fashion_model_flask.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
# load the weights into the model
model.load_weights('resources/fashion_model_flask.h5')

# Now we will create the flask api
# sta ring the flask application
app = Flask(__name__)


# Defining to classify image function


@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    # Define the uploads folder
    upload_dir = "resources/uploads/"
    # Load an uploaded image
    image = plt.imread(upload_dir + img_name)
    image.resize(28, 28)
    # Define the list of class names
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress",
               "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # Perform predictions with pre-trained model
    prediction = model.predict([image.reshape(1, 28*28)])

    class_index = np.argmax(prediction[0])
    print(f"Class Index Predicted - {class_index}")
    # return the prediction to the user
    return jsonify({"object_identified": classes[class_index]})


# Start the Flask application
app.run(port=5000, debug=False)

# %%
