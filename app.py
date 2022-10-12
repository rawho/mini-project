from flask import Flask, request, jsonify
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
import numpy as np
import pandas as pd
import requests

# import tensorflow
# from IPython.display import display, HTML
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
import pickle

app = Flask(__name__)

reconstructed_model = keras.models.load_model("cnn_model.hdf5", compile = True)

with open("meta.pkl","rb") as fh:
    cols,zscore_val,train_col,out_cols = pickle.load(fh)


def input_data(df):
    """
    data should be list of dictionary format
    """
   
    def encode_numeric_zscore_predict(df, name, mean=None, sd=None):
        # print(zscore_val[name])
        df[name] = (df[name] - zscore_val[name]["mean"]) / zscore_val[name]["sd"]

    def encode_text_dummy_predict(df, name):
        
        dummies = pd.get_dummies(df[name])
        for x in dummies.columns:
            dummy_name = f"{name}-{x}"
            df[dummy_name] = dummies[x]
        df.drop(name, axis=1, inplace=True)
        # Get missing columns in the training test
        
    for name in df.columns:
        if name == 'outcome':
            pass
        elif name in ['protocol_type','service','flag','land','logged_in',
                        'is_host_login','is_guest_login']:
            encode_text_dummy_predict(df,name)
        else:
            encode_numeric_zscore_predict(df,name)    

    # display 5 rows

    df.fillna(0)

    # Convert to numpy - Classification
    x_columns = df.columns#.drop('outcome')
    missing_cols = set( train_col ) - set( x_columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        df[c] = 0

    # print(df.head())
    # Ensure the order of column in the test set is in the same order than in train set

    #ToDo: if output is not presemt ,comment this line
    # dummies = pd.get_dummies(df['outcome'])
    
    df = df[train_col]
    
    x_columns = df.columns
    x = df[x_columns].values

    # dummies = pd.get_dummies(df['outcome']) # Classification
    #ToDo: if output is not presemt ,comment this line
    # outcomes = dummies.columns
    #ToDo: if output is not presemt ,comment this line
    # num_classes = len(outcomes)
    #ToDo: if output is not presemt ,comment this line
    # y = dummies.values
    #ToDo: if output is not presemt,returm only x
    return x#,y


@app.route('/predict', methods=['POST'])
def predict():

    # print(request.form.get("data")) 
    # data = [i for i in request.form.get("data").split(",")]
    # print(data)
    # print(type(data))

    # normal
     # data = [[0,"tcp","http","SF",215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]]

    # smurf
    data = [[0,"icmp","ecr_i","SF",1032,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,511,511,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,255,1.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00]]

    # neptune
    # data = [[0,"tcp","private","S0",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,238,19,1.00,1.00,0.00,0.00,0.08,0.05,0.00,255,19,0.07,0.05,0.00,0.00,1.00,1.00,0.00,0.00]]


    df = pd.DataFrame(data,columns=cols[:-1])

    x = input_data(df)
    
    out = reconstructed_model.predict(x)
    predict_class = [out_cols[np.argmax(each)] for each in out]

    return str(predict_class)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
