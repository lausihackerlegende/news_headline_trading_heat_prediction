import pandas as pd
import string
from nltk.tokenize import word_tokenize


# Preprocess a string of headlines for ML models
def cleaning(headlines: str) -> str:
    headlines = headlines.strip()
    headlines = headlines.lower()
    for punctuation in string.punctuation:
        headlines = headlines.replace(punctuation, "")
    tokens = word_tokenize(headlines)
    cleaned_headlines = " ".join(word for word in tokens)
    return cleaned_headlines


# Preprocess a string of headlines for DL (LSTM) models
def dl_preprocess(headlines: str) -> str:
    return cleaning(headlines)
