from flask import Flask
from tensorflow import keras
import pandas
import gzip
import math
import os
import re
import sys
from enum import Enum
from io import StringIO
from time import time

import matplotlib.pyplot as pyplot
import numpy
import pandas
import requests
import tensorflow
from IPython.display import display, HTML
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

app = Flask(__name__)



dataframe = pandas.DataFrame([[0,"tcp","http","SF",215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,"normal."]])


reconstructed_model = keras.models.load_model("cnn_model.hdf5", compile = False)
reconstructed_model.predict(dataframe)

@app.route('/predict', methods=['POST'])
def hello():
    return '<h1>Hello, World!</h1>'

if __name__ == "__main__":
    app.run(port=5000, debug=True)