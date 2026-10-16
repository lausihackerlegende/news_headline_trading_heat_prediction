"""
FastAPI application for serving the stock market weather prediction model.
"""
import os

import pandas as pd
from fastapi import FastAPI
from src.news_headline_trading_heat_prediction.predict import load_model, predict
from src.news_headline_trading_heat_prediction.pipeline import cleaning

MODEL_PATH ='models/lin_svc_pipeline.joblib'
#MODEL_PATH ='models/best_recall_model_logistic_regression.joblib'


# FastAPI instance
app = FastAPI()
app.state.model = load_model(MODEL_PATH)

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
def predict_weather(headline: str):
    headline = cleaning(headline)

    prediction = predict(app.state.model, headline)
    print(prediction)
    print(type(prediction[0]))
    #"It's going to be hot. Maybe. Or maybe cold. You never know"

    return {'prediction': float(prediction[0])}
