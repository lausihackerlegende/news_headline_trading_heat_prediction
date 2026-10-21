import os
import zipfile
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

DL_MODEL_PATH = "models/finbert_tone_heat_best.zip"

def _resolve_model_dir(extract_dir: str) -> str:
    """
    Finds the actual folder containing the model configuration.
    Sometimes when zipping a model, it creates an extra nested folder.
    This safety check ensures we point Hugging Face directly to the config.json file.
    """
    if os.path.exists(os.path.join(extract_dir, "config.json")):
        return extract_dir

    # If not in the root, check the immediate subdirectories
    for entry in os.listdir(extract_dir):
        candidate = os.path.join(extract_dir, entry)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            return candidate

    raise FileNotFoundError(f"Could not locate config.json in {extract_dir}")


def load_dl_model(model_path: str = DL_MODEL_PATH) -> dict:
    """
    Extracts the FinBERT-tone archive (if not already unpacked) and loads
    the model and tokenizer into memory. This happens once when the API starts.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Deep learning model artifact not found at: {model_path}")

    # 1. Handle Zip Extraction
    if model_path.endswith(".zip"):
        extract_dir = model_path.replace(".zip", "_extracted")
        # Only extract if the folder doesn't exist or is empty
        if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(model_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        load_dir = _resolve_model_dir(extract_dir)
    else:
        load_dir = _resolve_model_dir(model_path)

    # 2. Hardware Detection (Use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Load Tokenizer & Model
    tokenizer = BertTokenizer.from_pretrained(load_dir)
    model = BertForSequenceClassification.from_pretrained(load_dir).to(device)

    # Put the model in evaluation mode (disables dropout layers used only during training)
    model.eval()

    # 4. Extract Class Mappings from the model config
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "id2label": id2label,
        "label2id": label2id,
    }


def dl_predict(
    dl_bundle: dict,
    headline: str,
    summary: str = "",
) -> dict:
    """
    Runs FinBERT-tone inference and applies pure probability aggregation:
    Predicts 'hot' if the combined probability of (hot_rising + hot_falling) > cold.
    """
    tokenizer = dl_bundle["tokenizer"]
    model = dl_bundle["model"]
    device = dl_bundle["device"]
    id2label = dl_bundle["id2label"]
    label2id = dl_bundle["label2id"]

    # 1. Tokenize the input text into numerical IDs the model can understand
    # We pass both the headline and an optional summary.
    inputs = tokenizer(
        headline,
        summary if summary else '',
        truncation=True,
        max_length=512,
        return_tensors="pt", # Output PyTorch tensors
    ).to(device)

    # Disable gradient calculation to save memory and speed up inference
    with torch.no_grad():
        # 2. Forward Pass: Push the tokenized text through the neural network layers
        # The model returns raw, unnormalized mathematical scores called "logits"
        logits = model(**inputs).logits

        # 3. Convert Logits to Probabilities
        # https://huggingface.co/docs/transformers/model_doc/bert examples
        # Softmax forces the raw scores into a percentage distribution (adds up to 1.0)
        # dim=-1 ensures the math is applied across the 3 classes, not across batches
        # [0] unwraps PyTorch's default batch array so we get a flat list of 3 numbers
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # 4. Map the probabilities to their respective class IDs safely
    cold_id = label2id.get("cold", 0)
    rising_id = label2id.get("hot_rising", 1)
    falling_id = label2id.get("hot_falling", 2)

    p_cold = float(probs[cold_id])
    p_rising = float(probs[rising_id])
    p_falling = float(probs[falling_id])

    # Calculate the total "Hot" conviction by summing rising and falling
    p_hot_aggregated = p_rising + p_falling

    # 5. Business Logic: Predict hot if aggregated hot probability is strictly greater than cold
    is_hot = int(p_hot_aggregated > p_cold)

    # Identify which of the 3 classes had the single highest score natively
    native_3class_id = int(np.argmax(probs))

    # 6. Format the JSON response
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
