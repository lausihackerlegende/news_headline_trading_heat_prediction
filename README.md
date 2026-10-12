# 🌤️ Stock Market Weather: News-Driven Heat Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Le Wagon](https://img.shields.io/badge/Le%20Wagon-Data%20Science%20Batch%20%232361-orange.svg)](https://www.lewagon.com/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

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
- [Team](#team)
- [Credits](#credits)

---

## 🎯 About the Project
Can news headlines tell us if a stock is about to make an unusual move?

This project analyzes the relationship between financial news and stock price volatility. Instead of predicting exact price movements (which is notoriously noisy), we frame the problem as a binary classification task: predicting if a stock's daily price move will be significantly larger than its historical baseline ("Hot") or within normal parameters ("Cold").

---

## ️ The Mechanism: Hot vs. Cold
We define a "Hot" day dynamically for each stock:
1. **Usual Move**: Calculated as the standard deviation of the stock's daily percentage returns over the past 252 trading days (1 year).
2. **Hot Threshold**: A day is labeled **"Hot"** if the absolute daily price move is **> 1.5×** the usual move.
3. **Cold**: Everything else is labeled **"Cold"**.

*Example*: If Apple's (AAPL) usual daily move is 1.5%, and it moves 5.0% on a given day, that is 3.3× the usual move → **Hot**.

---

## 📊 Dataset
We use the **FNSPID (Financial News and Stock Price Integration Dataset)** (Dong et al., 2024).
- **Scope**: 15.7M news records, 4,775 companies (1999–2023).
- **Our Focus**: A filtered subset of NASDAQ-100 stocks for the year 2023.
- **Features (`X`)**: Aggregated daily news headlines and 3-sentence summaries (TextRank) joined by ticker and date.
- **Target (`y`)**: Binary label (`Hot` or `Cold`) based on the adjusted close price movement.

*Source*: [FNSPID on Hugging Face](https://huggingface.co/datasets/Zihan1004/FNSPID) | [arXiv Paper 2402.06698](https://arxiv.org/abs/2402.06698)

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
```text
news_headline_trading_heat_prediction/
├── notebooks/
│   └── fnspid-dataset-extraction.ipynb   # Dataset filtering and preprocessing
├── raw_data/
│   ├── nasdaq100_news_2023.parquet       # Filtered NASDAQ-100 news (2023)
│   └── nasdaq100_news_2023.csv.gz        # Compressed CSV version
├── src/
│   └── headline_heat/
│       ├── __init__.py
│       ├── data_processing.py            # Data loading and feature engineering
│       ├── features.py                   # TF-IDF and embedding pipelines
│       └── train.py                      # Model training and evaluation
├── pyproject.toml                        # Project metadata and dependencies (uv)
├── uv.lock                               # Locked dependency versions
── README.md
```

---

## ️ Installation & Setup

This project uses **[uv](https://github.com/astral-sh/uv)** for fast, reliable Python package management.

### Prerequisites
- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) installed

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lausihackerlegende/news_headline_trading_heat_prediction.git
   cd news_headline_trading_heat_prediction
   ```

2. **Create virtual environment and install dependencies with uv**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

   Or simply use:
   ```bash
   uv sync
   ```

3. **Data Preparation**:
   The pre-filtered NASDAQ-100 dataset for 2023 is already included in `raw_data/`.

   To extract your own subset from the full FNSPID dataset, run the notebook:
   ```bash
   jupyter notebook notebooks/fnspid-dataset-extraction.ipynb
   ```

---

##  Usage

### Train the Baseline Model
```bash
python src/headline_heat/train.py --model naive_bayes
```

### Run Data Exploration
```bash
jupyter notebook notebooks/fnspid-dataset-extraction.ipynb
```

### Interactive Demo (Coming Soon)
We plan to build a Streamlit app to demonstrate the model. Paste a stock's recent headlines to get a real-time "Hot" or "Cold" prediction with probability scores.
```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Evaluation Metrics
Because "Hot" days are inherently rare (imbalanced dataset), **Accuracy is a misleading metric**. We evaluate our models using:
- **Recall**: Of all truly "Hot" days, how many did we successfully catch?
- **Precision**: Of the days we predicted as "Hot", how many were actually "Hot"?
- **F1-Score**: The harmonic mean of Precision and Recall, providing a single robust metric to beat the "always predict Cold" baseline.

*Validation Strategy*: Strict chronological time-series split (e.g., Train: Jan-Sep 2023, Validation: Oct-Nov 2023, Test: Dec 2023) to prevent look-ahead bias and data leakage.

---

## 🔮 Future Roadmap
To elevate this project to production-grade quantitative finance standards, future iterations will include:
1. **RAG (Retrieval-Augmented Generation)**: Integrate a vector database to retrieve historical analogs of current headlines, allowing an LLM to explain *why* a day is predicted as "Hot".
2. **FinBERT Integration**: Replace TF-IDF/LSTM with domain-specific financial language models for superior sentiment and context understanding.
3. **Alternative Data**: Incorporate trading volume spikes and broader market regime indicators (e.g., VIX, SPY trends) as additional features.
4. **Backtesting Module**: Add a simulated portfolio performance chart showing returns of a strategy based on model predictions.
5. **Real-Time Pipeline**: Build an automated pipeline that fetches daily news headlines and generates predictions for the next trading day.

---

## 👥 Team
**Le Wagon Data Science Bootcamp - Batch #2361** (October 2026)

- **An Vo** ([An Vo](https://www.linkedin.com/in/an-vo-quant/))
- **Laurenz Hammacher**
- **Stanislav Korolev**

---

## 📚 Credits
- **Dataset**: FNSPID - Financial News and Stock Price Integration Dataset (Dong, Fan, & Peng, 2024)
- **Paper**: [FNSPID: A Comprehensive Financial News Dataset in Time Series](https://arxiv.org/abs/2402.06698) (KDD 2024 Applied Data Science Track Paper)
- **Bootcamp**: Le Wagon Data Science Bootcamp

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*⚠️ **Disclaimer**: This project is for educational and portfolio purposes only. It does not constitute financial advice. Do not use this model for real-world trading.*
