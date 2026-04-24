# 🎬 MediaStream — AI-Powered Movie Recommendation System

A **hybrid deep learning recommendation engine** that combines Neural Collaborative Filtering (NCF), TF-IDF content similarity, and language-aware boosting to deliver personalized, explainable movie recommendations across English, Hindi, and Telugu cinema.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green?logo=scikit-learn&logoColor=white)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Screenshots](#-screenshots)
- [Contributors](#-contributors)

---

## 🔍 Overview

**MediaStream** is a production-grade, language-aware movie recommendation system built as a deep learning case study. It solves the cold-start problem, supports multilingual content (English, Hindi, Telugu), and provides explainable AI-driven recommendations through a Netflix-inspired Streamlit dashboard.

### Key Highlights
- **5,400+ movies** spanning English, Hindi, and Telugu cinema (2000–2026)
- **122 users** with survey-based preference profiles
- **Hybrid scoring** combining deep learning + content similarity + language preferences
- **Explainable AI** — every recommendation comes with a reason

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   MediaStream Architecture                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│  │  User Survey │   │  Movie      │   │  Ratings        │   │
│  │  (DLDATA)    │   │  Catalog    │   │  (Interactions) │   │
│  └──────┬──────┘   └──────┬──────┘   └───────┬─────────┘   │
│         │                 │                   │              │
│         ▼                 ▼                   ▼              │
│  ┌──────────────────────────────────────────────────┐       │
│  │         Step 1: Data Preprocessing                │       │
│  │   • Label Encoding  • TF-IDF Vectorization       │       │
│  │   • Language Mapping • Genre Normalization        │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │         Step 2-3: NCF Model Training              │       │
│  │   User Embed(64) + Item Embed(64)                 │       │
│  │   → Dense(256) → Dense(128) → Dense(64) → σ(1)   │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │         Step 4: Hybrid Recommendation Engine      │       │
│  │   Score = 0.45×NCF + 0.30×Content + 0.25×LangBoost│      │
│  │   • Language-first filtering                      │       │
│  │   • Genre-strict matching                         │       │
│  │   • Cold-start fallback                           │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │         Step 5: Streamlit Dashboard               │       │
│  │   • Netflix-style glassmorphism UI                │       │
│  │   • User profiles & watch history                 │       │
│  │   • Real-time personalized recommendations        │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🧠 Deep Learning Engine
- **Neural Collaborative Filtering (NCF)** with 64-dim embeddings
- Multi-layer MLP with BatchNorm and Dropout for robust predictions
- Trained on 5,600+ user-movie interactions

### 🌐 Language-Aware Recommendations
- Detects preferred language from user watch history and survey data
- **Language-first filtering** ensures recommendations match user's language
- Supports **English**, **Hindi**, and **Telugu** cinema

### 📊 Hybrid Scoring Formula
```
Hybrid Score = 0.45 × NCF Score + 0.30 × Content Similarity + 0.25 × Language Boost
```
| Component | Weight | Description |
|-----------|--------|-------------|
| NCF Score | 45% | Deep learning collaborative filtering |
| Content Similarity | 30% | TF-IDF genre + language matching |
| Language Boost | 25% | Preferred language priority scoring |

### 🎯 Explainable AI
- Every recommendation includes a human-readable explanation
- Score breakdown (NCF, Content, Language) visible in the UI

### ❄️ Cold Start Support
- New users get recommendations based on genre/language preferences
- Survey-based fallback with popularity scoring

### 🎨 Premium Dashboard
- Netflix-inspired dark theme with glassmorphism effects
- Animated background orbs and micro-interactions
- Responsive movie cards with genre badges and score bars
- Interactive genre and language filters

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.12** | Core programming language |
| **TensorFlow / Keras** | NCF deep learning model |
| **scikit-learn** | TF-IDF vectorization, label encoding, train/test split |
| **Pandas / NumPy** | Data manipulation and preprocessing |
| **SciPy** | Sparse matrix operations, cosine similarity |
| **Streamlit** | Interactive web dashboard |

---

## 📁 Project Structure

```
CASE STUDY/
│
├── 📄 DLDATA.csv                    # User survey data (preferences, genres, languages)
├── 📄 movies_real_2000_2026.csv     # Telugu & Hindi movie catalog (2000-2026)
│
├── 🔧 step1_preprocess.py          # Data cleaning, encoding, TF-IDF matrix
├── 🧠 step2_model.py               # NCF model architecture definition
├── 🏋️ step3_train.py               # Model training with early stopping
├── ⚙️ step4_engine.py              # Hybrid recommendation engine
├── 🎨 step5_app.py                 # Streamlit dashboard UI
│
├── 🔄 rebuild_all.py               # Full pipeline rebuild (data → train → save)
│
├── 📊 generate_real_2000_2026.py   # Telugu/Hindi movie data generator
├── 📊 generate_balanced.py         # Balanced dataset generator
├── 📊 generate_hardcoded.py        # Hardcoded blockbuster data
│
├── 🔧 fix_genres_final.py          # Genre correction utilities
├── 🔧 fix_history.py               # Watch history alignment
├── 🔧 merge_new_movies.py          # Catalog merger utility
│
├── 🧪 test_all.py                  # Full test suite
├── 🧪 test_syntax.py               # Syntax validation
├── 🧪 test_fix.py                  # Fix validation tests
│
├── 🌐 collect_wiki.py              # Wikipedia genre scraper
├── 🌐 fetch_real_wikipedia.py      # Real Wikipedia data fetcher
├── 🌐 fetch_wiki.py                # Wiki API integration
│
└── 📄 .gitignore                   # Excludes model artifacts & caches
```

### Generated Artifacts (not in repo — regenerated via `rebuild_all.py`)
```
├── 📦 ncf_model.h5                 # Trained NCF model weights
├── 📦 user_encoder.pkl             # User label encoder
├── 📦 movie_encoder.pkl            # Movie label encoder
├── 📦 tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
├── 📦 content_matrix.npz           # Sparse TF-IDF content matrix
├── 📦 meta.pkl                     # Metadata (user/item counts, rating range)
├── 📄 movies_clean.csv             # Merged & cleaned movie catalog
├── 📄 ratings_clean.csv            # Processed ratings with encodings
└── 📄 cleaned_survey.csv           # Cleaned survey data
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Keerthimk24/media_personalized_recommendationsystem.git
cd media_personalized_recommendationsystem

# Install dependencies
pip install tensorflow pandas numpy scikit-learn scipy streamlit

# Place the DLCASE dataset folder at:
# c:\Users\navee\OneDrive\Desktop\DLCASE\
# (containing movies.csv and ratings.csv)
```

---

## ▶️ Usage

### Option 1: Full Pipeline Rebuild (First Time)
```bash
# Preprocesses data, merges catalogs, trains NCF model, saves all artifacts
python rebuild_all.py
```

### Option 2: Step-by-Step
```bash
# Step 1: Preprocess data
python step1_preprocess.py

# Step 2-3: Train the NCF model
python step3_train.py
```

### Launch the Dashboard
```bash
streamlit run step5_app.py
```

The app will open at `http://localhost:8501`

---

## 🧮 How It Works

### 1. Data Preprocessing (`step1_preprocess.py`)
- Loads user survey data (DLDATA.csv) with genre/language preferences
- Loads 5,000+ movies from the DLCASE catalog
- Merges 568 additional Telugu & Hindi movies (2000–2026)
- Maps ISO language codes → full names (en → English, te → Telugu, hi → Hindi)
- Encodes users/movies via `LabelEncoder` for neural network input
- Builds TF-IDF content vectors (genre + language weighted 3×)

### 2. NCF Model (`step2_model.py`)
```
Input: (user_id, movie_id) → Embeddings (64-dim each)
→ Concatenate (128-dim)
→ Dense(256, ReLU) + BatchNorm + Dropout(0.3)
→ Dense(128, ReLU) + BatchNorm + Dropout(0.25)
→ Dense(64,  ReLU) + BatchNorm + Dropout(0.2)
→ Dense(1, Sigmoid) → Predicted Rating [0, 1]
```

### 3. Training (`step3_train.py`)
- 80/20 train/test split
- Adam optimizer (lr=0.001) with binary crossentropy loss
- EarlyStopping (patience=5) + ReduceLROnPlateau
- 30 epochs, batch size 256

### 4. Hybrid Engine (`step4_engine.py`)
For each user request:
1. **Detect preferences** from watch history + survey (language, genres)
2. **Filter candidates** — language-first, then genre matching
3. **Score with NCF** — deep learning collaborative filtering
4. **Score with TF-IDF** — content similarity to watched movies
5. **Apply language boost** — reward matching language, penalize mismatch
6. **Rank & return** top-K with explainable score breakdown

### 5. Dashboard (`step5_app.py`)
- Netflix-style glassmorphism UI with animated backgrounds
- User selection with full profile display
- Watch history with star ratings
- Interactive genre/language filters
- Movie cards with match scores, genre badges, and AI explanations

---

## 📊 Dataset

| Dataset | Records | Source |
|---------|---------|--------|
| DLDATA.csv (Survey) | 122 users | Primary survey data |
| movies.csv (DLCASE) | 5,068 movies | MovieLens-style catalog |
| ratings.csv (DLCASE) | 5,675 ratings | User-movie interactions |
| movies_real_2000_2026.csv | 568 movies | Telugu (343) + Hindi (225) additions |
| **Merged Catalog** | **5,470 movies** | Combined & deduplicated |

### Language Distribution
| Language | Movies |
|----------|--------|
| English | 4,048 |
| Telugu | 364 |
| Hindi | 238 |
| Others | 820 |

---

## 🖼️ Screenshots

### Dashboard Home
> The Streamlit dashboard features a Netflix-inspired dark theme with glassmorphism effects, animated background orbs, and a premium movie recommendation interface.

### Recommendation Cards
> Each movie card displays the title, year, language tag, genre badges, match score bar, and an AI-generated explanation for why it was recommended.

---

## 👥 Contributors

- **Keerthimk24** — Full-stack development, deep learning model, recommendation engine, and UI design

---

## 📄 License

This project is part of an academic case study on personalized media recommendation systems using deep learning.

---

<div align="center">
  <b>Built with ❤️ using TensorFlow, scikit-learn, and Streamlit</b>
</div>
