"""
FastAPI application for serving the stock market weather prediction model.
"""
import os
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from src.news_headline_trading_heat_prediction.predict import load_model, predict
from src.news_headline_trading_heat_prediction.pipeline import cleaning
from src.news_headline_trading_heat_prediction.dl_predict import DL_MODEL_PATH, dl_predict, load_dl_model


MODEL_PATH ='models/lin_svc_pipeline.joblib'
MODEL_PATH_0_5 = 'models/lin_svc_pipeline_std_0_5.joblib'
MODEL_PATH_0_2 = 'models/lin_svc_pipeline_std_0_2.joblib'
#MODEL_PATH ='models/best_recall_model_logistic_regression.joblib'


# FastAPI instance
app = FastAPI(
    title="Stock Market Weather Prediction API",
    version="2.0.0",
)
app.state.baseline_model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
app.state.model_0_5 = load_model(MODEL_PATH_0_5)
app.state.model_0_2 = load_model(MODEL_PATH_0_2)
app.state.dl_bundle = load_dl_model(DL_MODEL_PATH) if os.path.exists(DL_MODEL_PATH) else None


class DLPayload(BaseModel):
    headline: str
    summary: Optional[str] = ""


# Root endpoint
@app.get("/")
def root():
    return {
        "greeting": "Hello investor",
        "baseline_model_loaded": app.state.baseline_model is not None,
        "dl_model_loaded": app.state.dl_bundle is not None,
    }


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
        prediction = predict(app.state.baseline_model, headline)
    elif model_choice == 0.5:
        prediction = predict(app.state.model_0_5, headline)
    elif model_choice == 0.2:
        prediction = predict(app.state.model_0_2, headline)



    #print(prediction)
    #print(type(prediction[0]))
    #"It's going to be hot. Maybe. Or maybe cold. You never know"

    return {'prediction': float(prediction[0])}


@app.get("/dl_predict")
def predict_weather_dl_get(
    headline: str = Query(..., description="Stock headline"),
    summary: str = Query("", description="3-sentence summary"),
):
    if not headline.strip():
        raise HTTPException(status_code=400, detail="Headline cannot be empty.")
    if app.state.dl_bundle is None:
        raise HTTPException(status_code=503, detail="Deep learning model is not loaded.")

    return dl_predict(
        dl_bundle=app.state.dl_bundle,
        headline=headline,
        summary=summary,
    )

# submitting information to a server so the server can process it
@app.post("/dl_predict")
def predict_weather_dl_post(payload: DLPayload): # must be in the DLPayload form
    if not payload.headline.strip(): # If there's not headline
        raise HTTPException(status_code=400, detail="Headline cannot be empty.")
    if app.state.dl_bundle is None: # if there's no model
        raise HTTPException(status_code=503, detail="Deep learning model is not loaded.")

    return dl_predict(
        dl_bundle=app.state.dl_bundle,
        headline=payload.headline,
        summary=payload.summary or "",
    )
