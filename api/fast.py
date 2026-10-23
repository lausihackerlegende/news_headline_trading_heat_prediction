"""FastAPI application for serving stock market weather models."""
import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.news_headline_trading_heat_prediction.predict import load_model, predict
from src.news_headline_trading_heat_prediction.pipeline import cleaning
from src.news_headline_trading_heat_prediction.dl_predict import (
    DL_MODEL_PATH, LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH,
    dl_predict, load_dl_model, load_lstm_model, lstm_predict
)

app = FastAPI(title="Stock Market Weather Prediction API", version="3.0.0")

# Safely load available models (Skipping missing ones without crashing)
app.state.baseline_model = load_model('models/lin_svc_pipeline.joblib') if os.path.exists('models/lin_svc_pipeline.joblib') else None
app.state.model_0_5 = load_model('models/lin_svc_pipeline_std_0_5.joblib') if os.path.exists('models/lin_svc_pipeline_std_0_5.joblib') else None
app.state.model_0_2 = load_model('models/lin_svc_pipeline_std_0_2.joblib') if os.path.exists('models/lin_svc_pipeline_std_0_2.joblib') else None

app.state.dl_bundle = load_dl_model(DL_MODEL_PATH) if os.path.exists(DL_MODEL_PATH) else None
app.state.lstm_bundle = load_lstm_model(LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH) if (os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_TOKENIZER_PATH)) else None


class Payload(BaseModel):
    headline: str
    summary: Optional[str] = ""


@app.get("/")
def root():
    return {
        "status": "Online",
        "svc_1_0_loaded": app.state.baseline_model is not None,
        "svc_0_5_loaded": app.state.model_0_5 is not None,
        "svc_0_2_loaded": app.state.model_0_2 is not None,
        "lstm_loaded": app.state.lstm_bundle is not None,
        "finbert_loaded": app.state.dl_bundle is not None,
    }

@app.get("/predict")
def predict_weather(headline: str, model_choice: int = 1):
    if not headline.strip():
        raise HTTPException(status_code=400, detail="Headline cannot be empty.")

    # Route model_choice=7 to LSTM
    if model_choice == 7:
        if not app.state.lstm_bundle:
            raise HTTPException(status_code=503, detail="LSTM model is not loaded on this server.")
        return lstm_predict(app.state.lstm_bundle, headline)

    if model_choice not in [1, 2, 5]:
        raise HTTPException(status_code=400, detail="Invalid choice. Use 1, 2, 5, or 7.")

    cleaned = cleaning(headline)

    # Select appropriate SVC model, fallback to baseline if specific threshold is missing
    model = app.state.baseline_model if model_choice == 1 else (
        app.state.model_0_5 or app.state.baseline_model if model_choice == 5 else app.state.model_0_2 or app.state.baseline_model
    )

    if not model:
        raise HTTPException(status_code=503, detail="SVC model is not loaded on this server.")

    return {"prediction": float(predict(model, cleaned)[0])}

@app.post("/lstm_predict")
def predict_weather_lstm_post(payload: Payload):
    if not payload.headline.strip():
        raise HTTPException(status_code=400, detail="Headline cannot be empty.")
    if not app.state.lstm_bundle:
        raise HTTPException(status_code=503, detail="LSTM model is not loaded on this server.")
    return lstm_predict(lstm_bundle=app.state.lstm_bundle, headline=payload.headline)

@app.post("/dl_predict")
def predict_weather_dl_post(payload: Payload):
    if not payload.headline.strip():
        raise HTTPException(status_code=400, detail="Headline cannot be empty.")
    if not app.state.dl_bundle:
        raise HTTPException(status_code=503, detail="FinBERT model is not loaded on this server.")
    return dl_predict(dl_bundle=app.state.dl_bundle, headline=payload.headline, summary=payload.summary)
