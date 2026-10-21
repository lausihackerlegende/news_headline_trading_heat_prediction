"""
FastAPI application for serving the stock market weather prediction model.
"""
import os
#import keras

import pandas as pd
from fastapi import FastAPI
from news_headline_trading_heat_prediction.predict import load_model, predict
from news_headline_trading_heat_prediction.pipeline import cleaning, dl_preprocess


MODEL_PATH = 'models/lin_svc_pipeline.joblib'
MODEL_PATH_0_5 = 'models/lin_svc_pipeline_std_0_5.joblib'
MODEL_PATH_0_2 = 'models/lin_svc_pipeline_std_0_2.joblib'
MODEL_PATH_it_13_w2v ='models/best_lstm_it13_w2v_frozen.joblib'


# FastAPI instance
app = FastAPI()
app.state.model = load_model(MODEL_PATH)
app.state.model_0_5 = load_model(MODEL_PATH_0_5)
app.state.model_0_2 = load_model(MODEL_PATH_0_2)
app.state.model_it_13_w2v = load_model(MODEL_PATH_it_13_w2v)

# Root endpoint
@app.get("/")
def root():
    """Root endpoint returning a greeting."""
    return {"greeting": "Hello investor"}


# Dummy endpoint
@app.get("/dummy")
def dummy(number: int):
    """Dummy endpoint that returns the square of the input number."""
    return {"result": number ** 2}


# predict endpoint
@app.get("/predict")
def predict_weather(headline: str, model_choice: int):
    lin_svm_model_range = [1, 5, 2]
    dl_model_range = [7, 8, 9]

    if model_choice in lin_svm_model_range:
        headline = cleaning(headline)
    elif model_choice in dl_model_range:
        headline = dl_preprocess(headline)

    if model_choice == 1:
        prediction = predict(app.state.model, headline)
    elif model_choice == 5:
        prediction = predict(app.state.model_0_5, headline)
    elif model_choice == 2:
        prediction = predict(app.state.model_0_2, headline)

    elif model_choice == 7:
        prediction = predict(app.state.model_it_13_w2v, headline)

    #print(prediction)
    #print(type(prediction[0]))
    #"It's going to be hot. Maybe. Or maybe cold. You never know"

    return {'prediction': float(prediction[0])}
