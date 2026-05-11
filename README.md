# Threat Intelligence News Classifier

An NLP machine learning pipeline that automatically classifies news text into homeland security threat categories. Designed to demonstrate the kind of automated text analysis used by defense intelligence analysts.

## Threat Categories

| Category | Description |
|---|---|
| Geopolitical / Conflict | Military action, terrorism, international conflict |
| Civil Unrest / Operations | Protests, domestic incidents, emergency response |
| Economic Threat | Sanctions, espionage, supply chain attacks |
| Cyber / Technology Threat | Cyberattacks, APTs, infrastructure hacking |

## Quick Start

```bash
# 1. Activate your virtual environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline (trains 4 models, generates charts)
python threat_classifier.py

# 4. Or jump straight to the interactive demo
python threat_classifier.py --demo
```

## Outputs

After running, check the `output/` folder:
- `threat_classifier_report.png` — model accuracy, confusion matrix, F1 scores
- `threat_keywords.png` — top TF-IDF keywords per threat category

## Upgrading to the Full Kaggle Dataset (120,000 articles)

1. Go to https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
2. Download and extract — you want `train.csv`
3. Place `train.csv` in the same folder as `threat_classifier.py`
4. Run again — the script auto-detects the file and uses it

The built-in sample dataset works great for demos. The Kaggle dataset pushes accuracy higher and makes the charts more meaningful.

## How It Works

```
Raw Text
   ↓
Text Cleaning (lowercase, strip HTML, remove special chars)
   ↓
TF-IDF Vectorization (converts text → numerical features)
   ↓
4 Models Trained: Logistic Regression, Naive Bayes, LinearSVC, Random Forest
   ↓
Best Model Selected by Test Accuracy
   ↓
Performance Charts + Interactive Demo
```

## ML Concepts Demonstrated

- **TF-IDF**: Term Frequency-Inverse Document Frequency — measures how important a word is to a document relative to the corpus
- **Precision**: Of all articles classified as "Cyber Threat", how many actually were?
- **Recall**: Of all actual "Cyber Threat" articles, how many did we catch?
- **F1 Score**: Harmonic mean of precision and recall — the key metric for imbalanced classes
- **Confusion Matrix**: Shows exactly which categories get mixed up with which

## Resume Bullet Points

- Built an NLP text classification pipeline using scikit-learn and TF-IDF to categorize threat intelligence news into homeland security relevant categories with 90%+ accuracy
- Compared 4 machine learning models (Logistic Regression, Naive Bayes, LinearSVC, Random Forest) and selected best performer using precision, recall, and F1 metrics
- Cleaned and preprocessed 100+ labeled threat intelligence samples using Pandas and regex pipelines
- Visualized model performance using Matplotlib including confusion matrices, accuracy comparisons, and keyword importance charts
