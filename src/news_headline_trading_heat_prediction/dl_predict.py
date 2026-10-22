import importlib.util
import os
import zipfile

# Force Keras 3 to use PyTorch to avoid a 2GB TensorFlow Docker bloat
if importlib.util.find_spec("tensorflow") is None:
    os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib
import keras
import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer

DL_MODEL_PATH = "models/finbert_tone_heat_best.zip"
LSTM_MODEL_PATH = "models/best_lstm_it13_w2v_frozen.keras"
LSTM_TOKENIZER_PATH = "models/tokenizer_min5.joblib"
LSTM_MAX_LEN = 83

def _resolve_model_dir(extract_dir: str) -> str:
    if os.path.exists(os.path.join(extract_dir, "config.json")):
        return extract_dir
    for entry in os.listdir(extract_dir):
        candidate = os.path.join(extract_dir, entry)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    raise FileNotFoundError(f"Could not locate config.json in {extract_dir}")

def load_dl_model(model_path: str = DL_MODEL_PATH) -> dict:
    if not os.path.exists(model_path):
        return None

    if model_path.endswith(".zip"):
        extract_dir = model_path.replace(".zip", "_extracted")
        if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(model_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        load_dir = _resolve_model_dir(extract_dir)
    else:
        load_dir = _resolve_model_dir(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(load_dir)
    model = BertForSequenceClassification.from_pretrained(load_dir).to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    return {"model": model, "tokenizer": tokenizer, "device": device, "id2label": id2label, "label2id": label2id}

def dl_predict(dl_bundle: dict, headline: str, summary: str = "") -> dict:
    tokenizer, model, device = dl_bundle["tokenizer"], dl_bundle["model"], dl_bundle["device"]
    id2label, label2id = dl_bundle["id2label"], dl_bundle["label2id"]

    inputs = tokenizer(headline, summary if summary else "", truncation=True, max_length=512, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    p_cold = float(probs[label2id.get("cold", 0)])
    p_rising = float(probs[label2id.get("hot_rising", 1)])
    p_falling = float(probs[label2id.get("hot_falling", 2)])
    p_hot_aggregated = p_rising + p_falling
    is_hot = int(p_hot_aggregated > p_cold)

    return {
        "prediction": is_hot,
        "label": "hot" if is_hot == 1 else "cold",
        "native_3class": id2label.get(int(np.argmax(probs)), "cold"),
        "prob_hot": round(p_hot_aggregated, 4),
        "probabilities": {"cold": round(p_cold, 4), "hot_rising": round(p_rising, 4), "hot_falling": round(p_falling, 4)}
    }

def load_lstm_model(model_path: str = LSTM_MODEL_PATH, tokenizer_path: str = LSTM_TOKENIZER_PATH) -> dict:
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None
    with open(tokenizer_path, "rb") as f:
        tokenizer = joblib.load(f)
    model = keras.saving.load_model(model_path, compile=False)
    return {"model": model, "tokenizer": tokenizer, "max_len": LSTM_MAX_LEN}

def lstm_predict(lstm_bundle: dict, headline: str) -> dict:
    tokenizer, model, max_len = lstm_bundle["tokenizer"], lstm_bundle["model"], lstm_bundle["max_len"]
    sequences = tokenizer.texts_to_sequences([headline.strip().lower()])
    padded = keras.utils.pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    raw_preds = np.asarray(model.predict(padded, verbose=0))
    p_hot = float(raw_preds.flat[0]) if raw_preds.shape[-1] == 1 else float(np.sum(raw_preds[0, 1:]))
    p_hot = float(np.clip(p_hot, 0.0, 1.0))
    p_cold = 1.0 - p_hot
    is_hot = int(p_hot >= 0.5)

    return {
        "prediction": is_hot,
        "label": "hot" if is_hot == 1 else "cold",
        "prob_hot": round(p_hot, 4),
        "prob_cold": round(p_cold, 4),
        "probabilities": {"cold": round(p_cold, 4), "hot": round(p_hot, 4)},
        "sequence_length": max_len,
        "matched_tokens": len(sequences[0]) if sequences and sequences[0] else 0,
    }
