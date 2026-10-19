import joblib
import pickle
import pandas as pd
from sklearn import set_config
set_config(transform_output="pandas")
from src.news_headline_trading_heat_prediction.pipeline import cleaning

MODEL_PATH ='models/lin_svc_pipeline.joblib'


def load_model(model_path):
    """load a trained model from the specifired file path"""

    return joblib.load(open(model_path, 'rb'))


def predict(model, data):
    """make predictions usding the loaded model and input data"""

    return model.predict([data])



if __name__ == "__main__":
    # Example usage
    model = load_model(MODEL_PATH)

    #Sample data
    sample_headline = 'Early IPO was a great success'

    # Run prediction
    prediction = predict(model, sample_headline)
    print("Prediction:", prediction)
