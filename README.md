# 🌤️ Stock Market Weather: News-Driven Heat Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Le Wagon](https://img.shields.io/badge/Le%20Wagon-Data%20Science%20Batch%20%232361-orange.svg)](https://www.lewagon.com/)

A machine learning project that predicts whether a tech stock will have a **"Hot"** (highly volatile) or **"Cold"** (normal) trading day based on aggregated daily news headlines. Built during the Le Wagon Data Science Bootcamp (Batch #2361).

---

## 📑 Table of Contents
- [About the Project](#about-the-project)
- [The Mechanism: Hot vs. Cold](#the-mechanism-hot-vs-cold)
- [Dataset](#dataset)
- [Modeling Approach](#modeling-approach)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Roadmap](#future-roadmap)
- [Team & Credits](#team--credits)

---

## 🎯 About the Project
Can news headlines tell us if a stock is about to make an unusual move?

This project analyzes the relationship between financial news and stock price volatility. Instead of predicting exact price movements (which is notoriously noisy), we frame the problem as a binary classification task: predicting if a stock's daily price move will be significantly larger than its historical baseline ("Hot") or within normal parameters ("Cold").

---

## 🌡️ The Mechanism: Hot vs. Cold
We define a "Hot" day dynamically for each stock:
1. **Usual Move**: Calculated as the standard deviation of the stock's daily percentage returns over the past 252 trading days (1 year).
2. **Hot Threshold**: A day is labeled **"Hot"** if the absolute daily price move is **> 1.5×** the usual move.
3. **Cold**: Everything else is labeled **"Cold"**.

*Example*: If Apple's (AAPL) usual daily move is 1.5%, and it moves 5.0% on a given day, that is 3.3× the usual move → **Hot**.

---

## 📊 Dataset
We use the **FNSPID (Financial News and Stock Price Integration Dataset)** (Dong et al., 2024).
- **Scope**: 15.7M news records, 4,775 companies (1999–2023).
- **Our Focus**: A filtered subset of ~25 major tech stocks (e.g., `AAPL`, `MSFT`, `NVDA`, `TSLA`, `META`, `GOOGL`).
- **Features (`X`)**: Aggregated daily news headlines (and optionally 3-sentence summaries) joined by ticker and date.
- **Target (`y`)**: Binary label (`Hot` or `Cold`) based on the adjusted close price movement.

*Source*: [FNSPID on Hugging Face](https://huggingface.co/datasets) | [arXiv Paper 2402.06698](https://arxiv.org/abs/2402.06698)

---

## 🧠 Modeling Approach
The project is developed in two progressive stages:

### Stage 1: Baseline (Week 1)
- **Text Processing**: TF-IDF (Term Frequency–Inverse Document Frequency) to score distinctive words in aggregated daily headlines.
- **Model**: Naive Bayes Classifier.
- **Goal**: Establish a baseline to learn which keywords or phrases correlate with volatile days.

### Stage 2: Deep Learning Stretch Goal (Week 2)
- **Text Processing**: Word Embeddings (e.g., Word2Vec or pre-trained financial embeddings).
- **Model**: Long Short-Term Memory (LSTM) network.
- **Goal**: Capture word order and context to not only predict "Hot" but also the direction ("Hot & Rising" vs. "Hot & Falling"). *E.g., "Apple beats expectations" vs. "Apple misses expectations".*

---

## 📁 Project Structure
*(Note: Adjust file names to match your actual repository)*
```text
news_headline_trading_heat_prediction/
├── data/
│   ├── raw/                  # Original FNSPID dataset extracts
│   └── processed/            # Filtered tech basket CSVs (X and y)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA and label distribution
│   ├── 02_baseline_model.ipynb     # TF-IDF + Naive Bayes
│   └── 03_lstm_stretch.ipynb       # Deep learning implementation
├── src/
│   ├── data_processing.py    # Scripts to join news and price data
│   ├── features.py           # TF-IDF and embedding pipelines
│   └── train.py              # Model training and evaluation scripts
├── app/
│   └── streamlit_app.py      # Interactive demo dashboard
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lausihackerlegende/news_headline_trading_heat_prediction.git
   cd news_headline_trading_heat_prediction
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Data Preparation**:
   Download the FNSPID dataset from Hugging Face, place it in the `data/raw/` folder, and run the preprocessing script to filter for the tech basket and generate the `X` (headlines) and `y` (hot/cold) tables.

---

## 🚀 Usage

### Train the Baseline Model
```bash
python src/train.py --model naive_bayes
```

### Run the Interactive Demo
We built a Streamlit app to demonstrate the model. Paste a stock's recent headlines to get a real-time "Hot" or "Cold" prediction with probability scores.
```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Evaluation Metrics
Because "Hot" days are inherently rare (imbalanced dataset), **Accuracy is a misleading metric**. We evaluate our models using:
- **Recall**: Of all truly "Hot" days, how many did we successfully catch?
- **Precision**: Of the days we predicted as "Hot", how many were actually "Hot"?
- **F1-Score**: The harmonic mean of Precision and Recall, providing a single robust metric to beat the "always predict Cold" baseline.

*Validation Strategy*: Strict chronological time-series split (e.g., Train: 2019–2021, Test: 2022–2023) to prevent look-ahead bias and data leakage.

---

## 🔮 Future Roadmap
To elevate this project to production-grade quantitative finance standards, future iterations will include:
1. **RAG (Retrieval-Augmented Generation)**: Integrate a vector database to retrieve historical analogs of current headlines, allowing an LLM to explain *why* a day is predicted as "Hot".
2. **FinBERT Integration**: Replace TF-IDF/LSTM with domain-specific financial language models for superior sentiment and context understanding.
3. **Alternative Data**: Incorporate trading volume spikes and broader market regime indicators (e.g., VIX, SPY trends) as additional features.
4. **Backtesting Module**: Add a simulated portfolio performance chart to the Streamlit app showing returns of a strategy based on model predictions.

---

## 👥 Team & Credits
- **Laurenz Hammacher** & Team (Le Wagon Data Science, Batch #2361, October 2026)
- **Dataset**: FNSPID (Dong et al., 2024)

*Disclaimer: This project is for educational and portfolio purposes only. It does not constitute financial advice. Do not use this model for real-world trading.*
