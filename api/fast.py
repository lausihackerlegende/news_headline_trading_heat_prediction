"""
FastAPI application for serving the stock market weather prediction model.
"""
import os

import pandas as pd
from fastapi import FastAPI
from src.news_headline_trading_heat_prediction.predict import load_model, predict
from src.news_headline_trading_heat_prediction.pipeline import cleaning


MODEL_PATH ='models/lin_svc_pipeline.joblib'
MODEL_PATH_0_5 = 'models/lin_svc_pipeline_std_0_5.joblib'
MODEL_PATH_0_2 = 'models/lin_svc_pipeline_std_0_2.joblib'
#MODEL_PATH ='models/best_recall_model_logistic_regression.joblib'


# FastAPI instance
app = FastAPI()
app.state.model = load_model(MODEL_PATH)
app.state.model_0_5 = load_model(MODEL_PATH_0_5)
app.state.model_0_2 = load_model(MODEL_PATH_0_2)

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
def predict_weather(headline: str, model_choice: float):
    headline = cleaning(headline)

    if model_choice == 1:
        prediction = predict(app.state.model, headline)
    elif model_choice == 0.5:
        prediction = predict(app.state.model_0_5, headline)
    elif model_choice == 0.2:
            prediction = predict(app.state.model_0_2, headline)



    #print(prediction)
    #print(type(prediction[0]))
    #"It's going to be hot. Maybe. Or maybe cold. You never know"

    return {'prediction': float(prediction[0])}
