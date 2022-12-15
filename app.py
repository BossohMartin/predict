import pickle
import pandas as pd
from flask import Flask, jsonify
import logging

DEBUG = True
app = Flask(__name__)

def read_dataframe(filename):
    data = pd.read_csv(filename)
    data.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1,inplace=True)
    data['SK_ID_CURR'] = data['SK_ID_CURR'].apply(int)
    data.drop(['TARGET'],axis=1,inplace=True)
    app.logger.info("succesfully loaded dataset:{}".format(filename))
    return data


def load_model(modelname):
    model = pickle.load(open("{}.pkl".format(modelname),'rb'))
    model.feature_name_
    model.set_params(n_jobs=1)
    app.logger.info("succesfully loaded model:{}".format(modelname))
    return model


model = load_model("LGBMClassifier")
data = read_dataframe('data_train_1.csv')

c_id = data['SK_ID_CURR'][0]
X = data[data['SK_ID_CURR'] == c_id]
logging.error("id:{}".format(c_id))
# check if the the id exist in the dataframe
if X.shape[0] == 1:
    classofinterest = 0 # depending if you want the positive or negative prediction
    y_proba = model.predict_proba(X)[0] # no need to do a predict and a predict proba
    proba = y_proba[classofinterest]
    y = proba.argmax() # by definition the max value of class probability is the predicted class

    result = {
                'prediction' : int(y),
                'proba' : float(proba),
                'id': c_id
                }
    logging.error("result:{}".format(result))

logging.error("done with setup")

@app.route('/')
def documentation():
    return jsonify('bienvenue dans mon application')

@app.route('/predict/<int:c_id>', methods=["GET"])
def predict_proba(c_id):
    X = data[data['SK_ID_CURR'] == c_id]
    # check if the the id exist in the dataframe
    if X.shape[0] == 1:
        model.classes_
        logging.info('enter predict')
        classofinterest = 0 # depending if you want the positive or negative prediction
        y_proba = model.predict_proba(X)[0] # no need to do a predict and a predict proba
        logging.info('end predict')
        proba = y_proba[classofinterest]
        y = proba.argmax() # by definition the max value of class probability is the predicted class

        result = {
                    'prediction' : int(y),
                    'proba' : float(proba),
                    'id': c_id
                    }
    else:
        result = False
    logging.info('jsonify')
    return jsonify(result)

@app.route('/myid/<myid>')
def myid(myid):
    return myid

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)