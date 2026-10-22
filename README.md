# 🌤️ Stock Market Weather: News-Driven Heat Prediction

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FinBERT--tone-yellow.svg)](https://huggingface.co/yiyanghkust/finbert-tone)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![Google Cloud Run](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4.svg)](https://cloud.google.com/run)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Le Wagon](https://img.shields.io/badge/Le%20Wagon-Data%20Science%20Batch%20%232361-orange.svg)](https://www.lewagon.com/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A quantitative machine learning system that predicts whether a stock will experience a **"Hot"** (abnormally volatile) or **"Cold"** (normal) trading day based on financial news headlines and text summaries. Developed during the Le Wagon Data Science Bootcamp (Batch #2361) and served via FastAPI containerized on Google Cloud Run.

---

## 📑 Table of Contents
- [About the Project](#-about-the-project)
- [The Mechanism: Hot vs. Cold](#-the-mechanism-hot-vs-cold)
- [Dataset](#-dataset)
- [Modeling Architecture](#-modeling-architecture)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [API Usage](#-api-usage)
- [Docker & GCP Cloud Run Deployment](#-docker--gcp-cloud-run-deployment)
- [Evaluation Performance](#-evaluation-performance)
- [Team & Credits](#-team--credits)

---

## 🎯 About the Project
Predicting directional asset price returns from news is notoriously noisy. This project reframes financial news analysis as an anomaly volatility classification problem: predicting whether a stock's daily price move will be greater than the 20-day rolling standard deviation of daily percentage returns for each asset.
2. **Hot Threshold**: A trading day is labeled **"Hot"** if absolute daily return **exceeds** its usual move and volume **exceeds** 20-day average volume.
3. **Cold**: All trading days within normal bounds are labeled **"Cold"**.

---

## 📊 Dataset
Built on the **FNSPID (Financial News and Stock Price Integration Dataset)** (Dong et al., 2024):
- **Scale**: Filtered subset of ~2.1M stock-days (`weather_all_stocks_master_dataset.parquet`).
- **Features (`X`)**: Headline (`news_title`) paired with 3-sentence TextRank summaries (`summary_3_sentences`).
- **Targets (`y`)**: 3-Class directional labels (`cold`, `hot_rising`, `hot_falling`) mapped to binary volatility states (`hot` vs `cold`). significantly exceed its historical baseline (**"Hot"**) or trade within standard bands (**"Cold"**).

---

## ⚡ The Mechanism: Hot vs. Cold
1. **Usual Move**: 20-day rolling standard deviation of daily percentage returns for each asset.
2. **Hot Threshold**: A trading day is labeled "Hot" if absolute daily return **exceeds** its usual move and volume **exceeds** 20-day average volume.
3. **Cold**: All trading days within normal bounds are labeled "Cold".

---

## 🧠 Modeling Architecture

### 1. Baseline Model (`LinearSVC`)
- Custom NLTK tokenization pipeline and TF-IDF representation.
- Fast, interpretable linear decision boundary for initial filtering (`models/lin_svc_pipeline.joblib`).

### 2. Deep Learning Model (`FinBERT-tone`)
- Fine-tuned `yiyanghkust/finbert-tone` using inverse class-weighted cross-entropy loss on 200,000 stratified financial samples (`models/finbert_tone_heat_best.zip`).
- **Probability Aggregation**:
  $$P(\text{Hot}) = P(\text{hot rising}) + P(\text{hot falling})$$
- **Decision Rule**: A day is classified as **Hot** strictly if the aggregated hot probability exceeds the cold probability: $P(\text{Hot}) > P(\text{Cold})$.

---

## 📁 Project Structure
```text
news_headline_trading_heat_prediction/
├── api/
│   └── fast.py                           # FastAPI application exposing /predict and /dl_predict
├── data/
│   └── processed/                        # Parquet datasets
├── models/
│   ├── lin_svc_pipeline.joblib           # Baseline TF-IDF + LinearSVC model
│   └── finbert_tone_heat_best.zip        # Fine-tuned FinBERT-tone weights
├── notebooks/
│   └── DL_model_evaluation.ipynb         # Stratified out-of-sample evaluation
├── src/
│   └── news_headline_trading_heat_prediction/
│       ├── __init__.py
│       ├── pipeline.py                   # NLTK text processing
│       ├── predict.py                    # Baseline LinearSVC inference
│       └── dl_predict.py                 # FinBERT inference and aggregation
├── Dockerfile                            # Optimized container recipe for Cloud Run
├── Makefile                              # Automation for local runs, builds, and GCP deploys
├── pyproject.toml                        # Project dependencies managed via uv
├── uv.lock                               # Locked dependency versions
└── README.md
```

---

## 🛠️ Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/lausihackerlegende/news_headline_trading_heat_prediction.git
cd news_headline_trading_heat_prediction

# 2. Install dependencies with uv
uv sync
uv run python -c "import nltk; nltk.download('punkt_tab')"
```

---

## 🚀 API Usage

### Local Execution
```bash
make run_api
# Or directly with uv:
uv run uvicorn api.fast:app --reload --port 8000
```
Interactive API docs are available at `http://127.0.0.1:8000/docs`.

### Baseline Prediction (`/predict`)
```bash
curl -G "http://127.0.0.1:8000/predict" \
  --data-urlencode "headline=Early IPO was a great success"
```

### Deep Learning Prediction (`/dl_predict`)
```bash
curl -X POST "http://127.0.0.1:8000/dl_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "headline": "Apple stock falls ahead of earnings despite bullish outlook",
    "summary": "Investors brace for heightened volatility ahead of quarterly report."
  }'
```
**Example Response:**
```json
{
  "prediction": 1,
  "label": "hot",
  "native_3class": "hot_falling",
  "prob_hot": 0.8642,
  "probabilities": {
    "cold": 0.1358,
    "hot_rising": 0.2104,
    "hot_falling": 0.6538
  }
}
```

---

## ☁️ Docker & GCP Cloud Run Deployment

Configure your `.env` file:
```bash
GCP_PROJECT=wagon-bootcamp-506511
GCP_REGION=europe-west1
DOCKER_REPO_NAME=trading-heat-repo
DOCKER_IMAGE_NAME=trading-heat-api
DOCKER_LOCAL_PORT=8000
GAR_MEMORY=4Gi
```

Deploy using Makefile automation:
```bash
# Set active GCP project
make gcloud-set-project

# Configure Artifact Registry authentication
make docker_allow

# Create the container repository (one-time setup)
make docker_create_repo

# Build linux/amd64 production image
make docker_build

# Push image to GCP Artifact Registry
make docker_push

# Deploy image to Google Cloud Run
make docker_deploy
```

---

## 📈 Evaluation Performance

On a held-out test split of 30,000 stratified stock-days:
- **3-Class Directional Classification**: Macro-F1 = `0.4672`.
- **Binary Volatility Classification**:
  - **Hot Recall**: **`83.6%`** 🔥
  - **Cold Recall**: `29.2%`
  - The model acts as a highly sensitive catalyst detector, successfully capturing and alerting on the vast majority of true volatile trading days.

---

## 🔮 Future Roadmap
1. **RAG (Retrieval-Augmented Generation)**: Retrieve historical news analogs from a vector database to explain *why* a stock-day is flagged as "Hot".
2. **Multimodal Market Features**: Combine FinBERT probability scores with pre-market volume spikes and implied volatility (VIX) regime indicators.
3. **Strategy Backtesting**: Simulate volatility-breakout and straddle strategies driven by `prob_hot` signals.

---

## 👥 Team & Credits
**Le Wagon Data Science Bootcamp - Batch #2361**
- **An Vo** ([LinkedIn](https://www.linkedin.com/in/an-vo-quant/))
- **Laurenz Hammacher**
- **Stanislav Korolev**

*Reference*: Dong, Z., Fan, C., & Peng, Z. (2024). *FNSPID: A Comprehensive Financial News Dataset in Time Series*. arXiv:2402.06698.
