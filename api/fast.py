"""
FastAPI application for serving the stock market weather prediction model.
"""
import os

import pandas as pd
from fastapi import FastAPI



# FastAPI instance
app = FastAPI()



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
def predict(headline: str):

    prediction = "It's going to be hot. Maybe. Or maybe cold. You never know"

    return prediction
