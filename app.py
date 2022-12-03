#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
from flask import Flask, jsonify, render_template, request
from lightgbm import LGBMClassifier


DEBUG = True
app = Flask(__name__)

def read_dataframe(filename):
    data = pd.read_csv(filename)
    data.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1,inplace=True)
    data['SK_ID_CURR'] = data['SK_ID_CURR'].apply(int)
    data.drop(['TARGET'],axis=1,inplace=True)
    return data


def load_model(modelname):
    modelname = 'LGBMClassifier'
    return pickle.load(open("LGBMClassifier.pkl",'rb'))


model = load_model(LGBMClassifier)
data = read_dataframe('data_train_1.csv')



@app.route('/')
def documentation():
    return jsonify('bienvenue dans mon application')


@app.route('/<ID_client>', methods=["POST", "GET"])
def predict_proba(ID_client):
    X = data[data['SK_ID_CURR'] == int(ID_client)]
    if X.shape[0] == 1:
        prediction = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0][prediction]

        dict_final = {
            'prediction' : int(prediction),
            'proba' : float(proba)
            }

        return jsonify(dict_final)
    else:
        return (False)
if __name__ == '__main__':
    app.run()




