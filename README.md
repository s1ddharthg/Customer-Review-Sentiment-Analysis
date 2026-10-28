# Customer Review Sentiment Analysis

A lightweight REST API that classifies customer reviews as **positive or negative** using a scikit-learn Decision Tree classifier and TF-IDF text features. Built with FastAPI, it provides a simple JSON interface for integrating sentiment predictions into other applications.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License">
</p>

<p align="center">
  <a href="https://github.com/s1ddharthg/Customer-Review-Sentiment-Analysis">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub Repository">
  </a>
  <img src="https://img.shields.io/badge/Model-Decision%20Tree-8A2BE2?style=flat-square" alt="Decision Tree">
  <img src="https://img.shields.io/badge/Features-TF--IDF-2EA44F?style=flat-square" alt="TF-IDF">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Model Details](#model-details)
- [Limitations](#limitations)
- [Deployment Considerations](#deployment-considerations)
- [License](#license)

## Overview

Customer Review Sentiment Analysis uses traditional machine learning and natural language processing to classify customer reviews into two sentiment categories.

The application transforms text into numerical features using **Term Frequency-Inverse Document Frequency (TF-IDF)** and passes those features to a trained **Decision Tree classifier**. A FastAPI service exposes the prediction pipeline through a REST endpoint that returns the predicted sentiment, its numeric representation, and a model confidence score when available.

The repository includes the trained model artifacts and a training pipeline for reproducing the classifier.

## Features

- **REST API:** Submit reviews through a JSON endpoint.
- **Binary sentiment classification:** Predict positive or negative sentiment.
- **TF-IDF vectorization:** Transform text into numerical features.
- **Confidence scores:** Return the predicted-class probability when supported by the classifier.
- **Interactive API documentation:** Test endpoints through Swagger UI and ReDoc.
- **Pretrained model artifacts:** Load the classifier and fitted vectorizer without retraining on every request.
- **Modular codebase:** Separate API, configuration, preprocessing, data download, and training logic.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| API framework | FastAPI |
| ASGI server | Uvicorn |
| Machine learning | scikit-learn |
| Feature extraction | TF-IDF |
| Classifier | Decision Tree |
| Model serialization | Joblib |
| Data processing | Pandas, NumPy |
| API validation | Pydantic through FastAPI |

## Project Structure

```text
Customer-Review-Sentiment-Analysis/
├── dataset/
│   └──                         # Training dataset
├── models/
│   ├── sentiment_model.pkl     # Trained classifier
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── notebooks/
│   └──                         # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── api.py                  # FastAPI application
│   ├── config.py               # Training configuration
│   ├── download.py             # Dataset download helper
│   ├── preprocessor.py         # Dataset preprocessing
│   └── train.py                # Model training pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- Dependencies listed in `requirements.txt`
- Trained model artifacts in the `models/` directory

### 1. Clone the Repository

```bash
git clone https://github.com/s1ddharthg/Customer-Review-Sentiment-Analysis.git
cd Customer-Review-Sentiment-Analysis
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

### 3. Start the API

Run the following command from the repository root:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

For local development with automatic reload:

```bash
python -m uvicorn src.api:app --reload
```

Once the server starts, access the following endpoints:

| Resource | URL |
|---|---|
| API root | [http://127.0.0.1:8000/](http://127.0.0.1:8000/) |
| Swagger UI | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) |
| ReDoc | [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) |
| OpenAPI schema | [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json) |

The application loads the serialized model artifacts during startup. Ensure both files exist and are compatible with the installed Python and scikit-learn environment.

## API Reference

### `GET /`

Returns basic service information.

**Example response**

```json
{
  "status": "Online",
  "endpoint": "/predict_sentiment"
}
```

This endpoint provides basic service information. It should not be treated as a comprehensive health check.

### `POST /predict_sentiment`

Classifies a customer review as positive or negative.

**Request body**

```json
{
  "review": "The product quality is excellent and shipping was fast!"
}
```

**Example response**

```json
{
  "review": "The product quality is excellent and shipping was fast!",
  "sentiment": 1,
  "label": "Positive",
  "confidence": 0.98
}
```

*The confidence value is illustrative. Actual output depends on the loaded model.*

#### Response Fields

| Field | Type | Description |
|---|---|---|
| `review` | String | Submitted review text, trimmed of surrounding whitespace |
| `sentiment` | Integer | `1` for positive, `0` for negative |
| `label` | String | Human-readable predicted sentiment |
| `confidence` | Number or `null` | Predicted-class probability, when supported |

#### HTTP Status Codes

| Code | Description |
|---|---|
| `200 OK` | Prediction successful |
| `422 Unprocessable Entity` | Invalid request or empty review |
| `500 Internal Server Error` | Prediction failed unexpectedly |
| `503 Service Unavailable` | Model artifacts unavailable |

These status codes describe the intended API behavior; verify the implementation for the exact error responses currently handled.

## Usage Examples

### cURL

Send a prediction request directly from your terminal:

```bash
curl -X POST "http://127.0.0.1:8000/predict_sentiment" \
  -H "Content-Type: application/json" \
  -d '{"review":"The product quality is excellent and shipping was fast!"}'
```

### Python

Use the API from another Python application with the `requests` library.

Install the client if needed:

```bash
python -m pip install requests
```

Make a request:

```python
import requests

url = "http://127.0.0.1:8000/predict_sentiment"

payload = {
    "review": "The product quality is excellent and shipping was fast!"
}

response = requests.post(
    url,
    json=payload,
    timeout=5,
)

response.raise_for_status()
result = response.json()

print("Sentiment:", result["label"])
print("Confidence:", result["confidence"])
```

### Interactive Testing

You can also test the endpoint without writing client code:

1. Start the API.
2. Open [Swagger UI](http://127.0.0.1:8000/docs).
3. Expand `POST /predict_sentiment`.
4. Select **Try it out**.
5. Enter a review and execute the request.
6. Inspect the JSON response.

## Model Details

### Pipeline

```text
Customer Review
       |
       v
Text Preprocessing
       |
       v
Fitted TF-IDF Vectorizer
       |
       v
Decision Tree Classifier
       |
       v
Predicted Sentiment
       |
       v
JSON API Response
```

The training workflow fits a TF-IDF vectorizer on the training text and uses the resulting features to train a Decision Tree classifier. Both artifacts are serialized with Joblib for reuse during inference.

### Label Mapping

The training labels are derived from star ratings:

| Rating | Sentiment | Numeric Label |
|---|---|---|
| 4 or higher | Positive | `1` |
| Below 4 | Negative | `0` |

The classifier learns from these derived labels. Its output represents a prediction based on patterns in the training data, not a verified measure of customer satisfaction or product quality.

### Model Artifacts

| File | Purpose |
|---|---|
| `models/sentiment_model.pkl` | Trained Decision Tree classifier |
| `models/tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer used to transform text |

The vectorizer used at inference must be the same fitted artifact, or a compatible artifact with the exact same learned feature vocabulary and ordering, that was used to train the classifier.

## Limitations

- **Title-based training:** The current model is trained using the `Review Title` field. Its performance on complete review bodies has not been established.
- **Domain dependence:** Results depend on the training dataset and may not generalize to other products, languages, or review domains.
- **Probability calibration:** A Decision Tree's predicted-class probability is not necessarily a calibrated measure of reliability.
- **Binary output:** The model does not distinguish neutral, mixed, or nuanced sentiment as separate classes.
- **Limited contextual understanding:** Traditional TF-IDF features do not capture language context and meaning in the same way as contextual language models.
- **Input consistency:** Training and inference must use compatible preprocessing and text fields.

## Deployment Considerations

Before exposing the API to public traffic, consider the following:

- Evaluate the model on a held-out test set using precision, recall, F1-score, and a confusion matrix.
- Add automated tests for successful predictions, malformed requests, empty reviews, and missing model artifacts.
- Pin dependencies and keep training and inference environments compatible.
- Load only trusted serialized model files because untrusted pickle-based artifacts can execute malicious code when loaded.
- Configure structured logging, monitoring, and appropriate exception handling.
- Add request-size limits, rate limiting, and access controls where necessary.
- Use HTTPS when deployed publicly.
- Configure separate liveness and readiness checks.
- Disable automatic reload mode in production.

**Production status:** This repository is a runnable inference example, not a fully hardened production service.

## License

Distributed under the [MIT License](LICENSE).

---

<p align="center">
  Built with Python, FastAPI, and scikit-learn.
</p>