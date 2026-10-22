import os
import zipfile
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

DL_MODEL_PATH = "models/finbert_tone_heat_best.zip"


def _resolve_model_dir(extract_dir: str) -> str:
    """Find the directory containing config.json if the zip has a nested folder."""
    if os.path.exists(os.path.join(extract_dir, "config.json")):
        return extract_dir
    for entry in os.listdir(extract_dir):
        candidate = os.path.join(extract_dir, entry)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    raise FileNotFoundError(f"Could not locate config.json in {extract_dir}")

# Load the model and tokenizer
def load_dl_model(model_path: str = DL_MODEL_PATH) -> dict:
    """
    Extracts the FinBERT-tone archive (if not already unpacked) and loads
    the model and tokenizer into memory.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Deep learning model artifact not found at: {model_path}")

    if model_path.endswith(".zip"):
        extract_dir = model_path.replace(".zip", "_extracted")
        # Make a new directory to extract the model
        if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(model_path, "r") as zip_ref: # read only
                zip_ref.extractall(extract_dir)
        # Fine the config.json file in the extract folder
        load_dir = _resolve_model_dir(extract_dir)
    else:
        load_dir = _resolve_model_dir(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(load_dir)
    model = BertForSequenceClassification.from_pretrained(load_dir).to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "id2label": id2label,
        "label2id": label2id,
    }


# Use the loaded model to predict
def dl_predict(
    dl_bundle: dict,
    headline: str,
    summary: str = "",
) -> dict:
    """
    Runs FinBERT-tone inference and applies pure probability aggregation:
    Predicts 'hot' if P(hot_rising) + P(hot_falling) > P(cold)
    """
    tokenizer = dl_bundle["tokenizer"]
    model = dl_bundle["model"]
    device = dl_bundle["device"]
    id2label = dl_bundle["id2label"]
    label2id = dl_bundle["label2id"]

    inputs = tokenizer(
        headline,
        summary if summary else "",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        # If there's a tie, take the first value
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    cold_id = label2id.get("cold", 0)
    rising_id = label2id.get("hot_rising", 1)
    falling_id = label2id.get("hot_falling", 2)

    p_cold = float(probs[cold_id])
    p_rising = float(probs[rising_id])
    p_falling = float(probs[falling_id])
    p_hot_aggregated = p_rising + p_falling

    # Predict hot if aggregated hot probability is greater than cold
    is_hot = int(p_hot_aggregated > p_cold)
    native_3class_id = int(np.argmax(probs))

    return {
        "prediction": is_hot,
        "label": "hot" if is_hot == 1 else "cold",
        "native_3class": id2label.get(native_3class_id, "cold"),
        "prob_hot": round(p_hot_aggregated, 4),
        "probabilities": {
            "cold": round(p_cold, 4),
            "hot_rising": round(p_rising, 4),
            "hot_falling": round(p_falling, 4),
        },
    }
