# Threat Intelligence News Classifier

NLP ML pipeline that classifies news text into homeland security threat categories. Designed to demonstrate the kind of automated text analysis used by defense intelligence analysts.

## Threat Categories

| Category | Description |
|---|---|
| Geopolitical / Conflict | Military action, terrorism, international conflict |
| Civil Unrest / Operations | Protests, domestic incidents, emergency response |
| Economic Threat | Sanctions, espionage, supply chain attacks |
| Cyber / Technology Threat | Cyberattacks, APTs, infrastructure hacking |

## Quick Start

```bash
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

python threat_classifier.py

python threat_classifier.py --demo
```

## Outputs

After running, check `output/` folder.
- `threat_classifier_report.png` shows the model accuracy, confusion matrix, F1 scores.
- `threat_keywords.png` shows top TF-IDF keywords per threat category

## Using the Full Kaggle Dataset (120,000 articles)

1. Go to https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
2. Download and extract, since you want `train.csv`
3. Place `train.csv` in the same folder as `threat_classifier.py`
4. Run again since the script auto-detects the file and uses it

The built-in sample dataset works great for demos, yet the Kaggle dataset pushes accuracy higher and makes the charts more meaningful and applicable to more real-world scenarios.

## How It Works

```
- Raw Text

- Cleaning (lowercase, strip HTML, remove special chars)
   
- TF-IDF Vectorization (converts text to numerical features)
   ↓
- 4 Models are Trained (Logistic Regression, Naive Bayes, LinearSVC, Random Forest)
   ↓
- The Best Model Selected by Test Accuracy

- Performance Charts + Interactive Demo
```

## ML Concepts Learned From This

- **TF-IDF**: Term Frequency-Inverse Document Frequency — measures how important a word is to a document relative to the corpus
- **Precision**: Of all articles classified as "Cyber Threat", how many actually were?
- **Recall**: Of all actual "Cyber Threat" articles, how many did we catch?
- **F1 Score**: Harmonic mean of precision and recall — the key metric for imbalanced classes
- **Confusion Matrix**: Shows exactly which categories get mixed up with which
