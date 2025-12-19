**# 🚀 Amazon ML Challenge 2025 — Smart Product Pricing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![Platform](https://img.shields.io/badge/Platform-Windows%20|%20macOS%20|%20Linux-lightgrey)]()  
![Status](https://img.shields.io/badge/Status-Completed-green.svg)  
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📖 Table of Contents
- [Project Description](#-project-description)  
- [Key Features](#-key-features)  
- [Repository Structure](#-repository-structure)  
- [Environment Setup](#-environment-setup)  
- [Data Preparation](#-data-preparation)  
- [Quick Start](#-quick-start)  
- [Approach Overview](#-approach-overview)  
- [Evaluation](#-evaluation)  
- [Artifacts & Outputs](#-artifacts--outputs)  
- [Reproducibility](#-reproducibility)  
- [Troubleshooting](#-troubleshooting)  
- [Acknowledgements](#-acknowledgements)  
- [License](#-license)  

---

## 📖 Project Description
**Smart Product Pricing** is a multimodal machine learning solution for the **Amazon ML Challenge 2025**. It predicts product prices by integrating:

- **Text**: Product catalog content  
- **Images**: Product images  
- **Numerical Features**: Engineered metrics like value/unit, content statistics, brand heuristics  

The system combines **Fusion MLP**, **LightGBM**, and **XGBoost** base models with a **Ridge meta-learner** to produce final predictions optimized for **SMAPE**.

---

## 🌟 Key Features

### ⚡ Core Functionality
| Feature | Implementation |
|---------|----------------|
| **Multimodal Fusion** | CLIP text & image embeddings + engineered numeric features |
| **Base Models** | PyTorch Fusion MLP, LightGBM, XGBoost |
| **Stacking** | Ridge regression meta-learner using OOF predictions |
| **Optimized Training** | PCA, TF-IDF SVD, caching, and Optuna tuning |
| **Hardware Support** | CPU, macOS MPS, CUDA GPUs |

### 📊 Outputs
- Submission-ready `test_out.csv`  
- Model artifacts: PCA objects, scalers, TF-IDF SVD, OOF logs  

---

## 📂 Repository Structure
.
├── preprocess.py # Feature engineering & train/test preprocessing
├── embed_text.py # CLIP text embeddings
├── embed_images.py # CLIP image embeddings with caching
├── train.py # Base models + stacking + submission
├── src/utils.py # Utilities (image download, caching)
├── dataset/ # Place train.csv and test.csv here
├── artifacts/ # Saved PCA, scalers, OOF logs, meta-model inputs
└── README.md

---

## 🛠️ Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Install core dependencies
pip install pandas numpy scikit-learn lightgbm xgboost optuna tqdm pillow requests joblib pyarrow

# Install PyTorch
pip install torch torchvision torchaudio  # CPU/MPS compatible

# Install OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

macOS Apple Silicon automatically uses MPS; CUDA GPUs supported.
```

📦 Data Preparation

Place files in dataset/train.csv and dataset/test.csv

Required columns: sample_id, catalog_content, image_link, price (train only)

⚡ Quick Start
# Preprocess train/test into combined Parquet
python preprocess.py --train dataset/train.csv --test dataset/test.csv --out preprocessed.parquet

# Generate CLIP text embeddings
python embed_text.py

# Generate CLIP image embeddings (with caching)
python embed_images.py

# Train models, stack predictions, produce submission
python train.py --trials 20 --mlp_epochs 12 --pca_text 128 --pca_img 128
Outputs:

test_out.csv — submission file

Model artifacts — PCA/scalers, TF-IDF SVD, OOF logs, meta-model inputs

📝 Approach Overview
Text Features

CLIP ViT-B/32 embeddings → normalized → PCA

TF-IDF bigram features → TruncatedSVD for tree models

Image Features

CLIP ViT-B/32 embeddings → normalized → PCA

Numerical Features

Value/unit normalization, content length, word counts, ratios/log transforms

Brand heuristics & target encoding

Base Learners

Fusion MLP over [text_pca, image_pca, numeric] blocks

LightGBM and XGBoost with Optuna hyperparameter tuning

Stacking

Concatenate OOF predictions → Ridge regression meta-model → final test predictions → clip positive → test_out.csv

📊 Evaluation

Metric: SMAPE (Symmetric Mean Absolute Percentage Error)

Prints per-fold OOF SMAPE for MLP, LightGBM, and XGBoost
## 📦 Artifacts & Outputs

| Artifact            | Description                           |
|--------------------|---------------------------------------|
| `pca_text.pkl`      | PCA object for text embeddings        |
| `pca_img.pkl`       | PCA object for image embeddings       |
| `scaler_*.pkl`      | Normalization scalers                 |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer                  |
| `tfidf_svd.pkl`     | TF-IDF SVD reducer                    |
| `imputer.pkl`       | Missing value imputer                 |
| `oof_*.pkl`         | Out-of-fold predictions               |
| `test_preds_*.pkl`  | Base model test predictions           |

## 🔁 Reproducibility

- **Seeds**: `RANDOM_SEED=42` for NumPy & PyTorch  
- **Optuna**: Fixed TPE seed; minor nondeterminism may remain  
- **Hardware**: CPU/MPS/CUDA differences may slightly affect results  

---

## 🛠️ Troubleshooting

- **CLIP install issues**: Ensure `git` is available  
- **Image download throttling**: `embed_images.py` caches images; use `src/utils.py` to pre-download  
- **MPS memory errors**: Reduce `BATCH_SIZE` in `embed_images.py`  

---

## 🙏 Acknowledgements

- Amazon ML Challenge 2025 organizers  
- OpenAI CLIP for text & image embeddings  
- LightGBM & XGBoost libraries  
- Optuna for hyperparameter optimization  

---

## 📄 License

Project license is currently unspecified. Please define before public distribution.**
