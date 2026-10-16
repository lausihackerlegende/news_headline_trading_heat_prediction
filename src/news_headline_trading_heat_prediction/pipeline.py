import pandas as pd
import string
from nltk.tokenize import word_tokenize

def cleaning(headlines):

    headlines = headlines.strip()
    headlines = headlines.lower()
    #headlines = ''.join(char for char in headlines if not char.isdigit())
    for punctuation in string.punctuation:
        headlines = headlines.replace(punctuation, '')
    tokens = word_tokenize(headlines)
    # stop_words = set(stopwords.words('english'))
    #tokens_cleaned = [w for w in tokens if not w in stop_words]
    #lemmatized = [WordNetLemmatizer().lemmatize(word, pos="v") for word in tokens]
    cleaned_headlines = ' '.join(word for word in tokens)
    return cleaned_headlines
