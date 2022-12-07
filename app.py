from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder="templates")
train_df = pd.read_csv("imdb_train.csv")
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
vectorizer.fit_transform(train_df.review)


def ValuePredictor(to_predict_list):
    to_predict = vectorizer.transform(to_predict_list)
    loaded_model = pickle.load(open("bayn_model.pkl", "rb"))
    # to_predict = vectorizer.transform(to_predict)
    result = loaded_model.predict(to_predict)

    return result[0]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        print(to_predict_list)
        to_predict_list = list(to_predict_list.values())
        print(to_predict_list)
        to_predict_list = list(map(str, to_predict_list))
        print(to_predict_list)
        result = ValuePredictor(to_predict_list)

        if int(result) == 1:
            prediction = 'Positif'

        else:
            prediction = 'Negatif'

        return render_template("result.html", phrase=(str, to_predict_list), prediction=prediction)
