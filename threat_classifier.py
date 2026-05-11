"""
============================================================
  Threat Intelligence News Classifier
  ─────────────────────────────────────────────────────────
  Automatically classifies news text into threat categories
  relevant to homeland security and defense analysis.

  Stack: Pandas · NumPy · Scikit-learn · Matplotlib
  Dataset: AG News (via Kaggle) — 120,000 news articles

  Run:
      python threat_classifier.py              ← full pipeline
      python threat_classifier.py --demo       ← interactive demo only
============================================================
"""

import sys
import os
import re
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — saves PNGs instead of popup windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

# ─── Constants ────────────────────────────────────────────────────────────────

# AG News original labels → mapped to homeland security threat categories
# AG News classes: 1=World, 2=Sports, 3=Business, 4=Sci/Tech
LABEL_MAP = {
    1: "Geopolitical / Conflict",   # World news → geopolitical threats
    2: "Civil Unrest / Operations", # Sports → repurposed for civil events
    3: "Economic Threat",           # Business → economic/financial threats
    4: "Cyber / Technology Threat", # Sci/Tech → cyber threats
}

# Color palette — dark military green theme
COLORS = {
    "Geopolitical / Conflict":   "#c0392b",   # red
    "Civil Unrest / Operations": "#e67e22",   # orange
    "Economic Threat":           "#f1c40f",   # yellow
    "Cyber / Technology Threat": "#27ae60",   # green
}
PALETTE = list(COLORS.values())

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── 1. Data Loading ──────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load AG News dataset.
    Tries local CSV first (if you downloaded from Kaggle),
    then falls back to a reliable built-in sample so the script
    always runs without needing a Kaggle account.
    """
    local_paths = [
        "train.csv",
        "ag_news_train.csv",
        os.path.join("data", "train.csv"),
    ]

    for path in local_paths:
        if os.path.exists(path):
            print(f"[INFO] Loading dataset from: {path}")
            df = pd.read_csv(path)
            df = df.rename(columns={"Class Index": "label", "Title": "title", "Description": "description"})
            df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
            df["threat_category"] = df["label"].map(LABEL_MAP)
            df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
            df["threat_category"] = df["label"].map(LABEL_MAP)
            print(f"[INFO] Loaded {len(df):,} articles from Kaggle CSV")
            return df

    # ── Built-in sample dataset (always works, no download needed) ─────────────
    print("[INFO] No Kaggle CSV found — using built-in threat intelligence dataset.")
    print("[INFO] (See README for how to add the full 120k Kaggle dataset)\n")

    samples = [
        # Geopolitical / Conflict
        (1, "US military conducts airstrikes against militant targets in Syria"),
        (1, "NATO forces increase presence near eastern European borders"),
        (1, "Terrorist group claims responsibility for embassy bombing"),
        (1, "UN Security Council convenes emergency session over nuclear threat"),
        (1, "Armed conflict intensifies as rebel forces advance on capital city"),
        (1, "Diplomatic tensions escalate as ambassador recalled following attack"),
        (1, "Border skirmish leaves dozens dead in disputed territory"),
        (1, "Sanctions imposed on rogue state over weapons program violations"),
        (1, "Intelligence agencies warn of imminent attack on Western interests"),
        (1, "Peace talks collapse as ceasefire violations reported across region"),
        (1, "Special forces deployed to counter insurgency in northern provinces"),
        (1, "Chemical weapons inspectors denied access to disputed facilities"),
        (1, "Hostage situation ends as security forces storm compound"),
        (1, "Refugee crisis deepens as conflict forces mass displacement"),
        (1, "Drone strike eliminates senior terror organization leadership figure"),
        (1, "Joint military exercises signal alliance commitment to region defense"),
        (1, "Ballistic missile test condemned by international community leaders"),
        (1, "Arms embargo violated as weapons shipment intercepted at sea"),
        (1, "Assassination attempt on foreign minister raises regional tensions"),
        (1, "Underground nuclear facility detected by satellite imagery analysis"),
        (1, "Counter-terrorism unit dismantles bomb-making operation in city"),
        (1, "Extremist group recruits fighters through encrypted online platforms"),
        (1, "Military coup topples government, martial law declared nationwide"),
        (1, "Biological weapons cache discovered in abandoned warehouse facility"),
        (1, "Coalition forces push back against territorial gains by militants"),

        # Civil Unrest / Operations
        (2, "Protesters clash with police outside parliament in capital city"),
        (2, "Mass demonstration turns violent as tear gas deployed by officers"),
        (2, "Riot police mobilized as civil unrest spreads to multiple cities"),
        (2, "Government declares state of emergency following widespread protests"),
        (2, "Community organizing efforts focus on police accountability measures"),
        (2, "Labor strike paralyzes transportation network across major region"),
        (2, "Public health emergency declared after contamination event detected"),
        (2, "Disaster relief operations underway following major flooding event"),
        (2, "Emergency responders coordinate evacuation of coastal communities"),
        (2, "Search and rescue teams deployed after building collapse downtown"),
        (2, "National Guard activated to assist with wildfire containment efforts"),
        (2, "FEMA coordinates response to hurricane damage across gulf states"),
        (2, "Curfew enforced as looting reported in affected neighborhoods"),
        (2, "Humanitarian aid corridors established for displaced population"),
        (2, "Counter-protest movement gathers in response to extremist rally"),
        (2, "Food supply disruption triggers unrest in urban population centers"),
        (2, "Water treatment facility sabotage causes public health emergency"),
        (2, "Anti-government militia seizes federal building in standoff"),
        (2, "Crisis negotiators deployed to resolve armed standoff situation"),
        (2, "Power grid failure leaves millions without electricity regionally"),
        (2, "Infrastructure attack disrupts fuel supply to major urban areas"),
        (2, "Gang violence escalates prompting emergency law enforcement response"),
        (2, "Extremist propaganda fuels recruitment in vulnerable communities"),
        (2, "Domestic terrorist plot foiled by joint law enforcement operation"),
        (2, "Supply chain attack detected in critical government contractor systems"),

        # Economic Threat
        (3, "Sanctions cripple economy as currency collapses to record low"),
        (3, "Foreign state accused of manipulating financial markets covertly"),
        (3, "Defense contractor faces scrutiny over foreign ownership concerns"),
        (3, "Oil supply disruption sends energy prices to decade-long highs"),
        (3, "Economic espionage ring dismantled after months-long FBI operation"),
        (3, "Critical supply chain vulnerability exposed in semiconductor sector"),
        (3, "Foreign investment in strategic port raises national security alarm"),
        (3, "Money laundering network used to fund terrorist activities dismantled"),
        (3, "Ransomware attack demands millions from critical infrastructure firm"),
        (3, "Trade war escalates with new tariffs targeting defense supply chains"),
        (3, "Financial fraud scheme funnels funds to sanctioned organizations"),
        (3, "Hostile nation acquires controlling stake in sensitive technology firm"),
        (3, "Cryptocurrency used to evade sanctions imposed by Western nations"),
        (3, "Pentagon warns of dependency on adversary-controlled rare earth minerals"),
        (3, "Bank system targeted in coordinated cyber-enabled financial attack"),
        (3, "Counterfeit goods network funds terrorist organization operations"),
        (3, "Strategic petroleum reserve tapped as fuel prices spike regionally"),
        (3, "Export control violations result in indictment of technology firm"),
        (3, "Intellectual property theft costs defense sector billions annually"),
        (3, "Dark web marketplace facilitates weapons and drug trafficking globally"),
        (3, "Foreign government subsidizes competitors undermining US industry"),
        (3, "Pension fund infiltrated by foreign actors seeking sensitive data"),
        (3, "Energy infrastructure deal blocked over national security concerns"),
        (3, "Black market weapons sales traced to sanctioned state actor network"),
        (3, "Financial sanctions enforcement tightened against terror financing"),

        # Cyber / Technology Threat
        (4, "Nation-state hackers breach defense contractor network stealing data"),
        (4, "Critical infrastructure attacked by sophisticated malware campaign"),
        (4, "Zero-day exploit used to infiltrate government agency systems widely"),
        (4, "Ransomware cripples hospital network disrupting emergency services"),
        (4, "APT group linked to foreign intelligence service targets Pentagon"),
        (4, "Spear-phishing campaign compromises senior officials email accounts"),
        (4, "CISA issues emergency directive after widespread software vulnerability"),
        (4, "Power grid control systems targeted by coordinated cyber intrusion"),
        (4, "Deepfake video used in disinformation campaign ahead of election"),
        (4, "Satellite communications disrupted by suspected jamming operation"),
        (4, "Classified data exfiltrated through compromised insider threat actor"),
        (4, "AI-generated propaganda floods social media before key election"),
        (4, "Supply chain attack embeds backdoor in widely-used software package"),
        (4, "Water treatment plant SCADA system accessed by unauthorized attacker"),
        (4, "Drone swarm demonstrates new asymmetric warfare capability threat"),
        (4, "Facial recognition database breached exposing intelligence officer identities"),
        (4, "5G infrastructure concerns raised over foreign equipment manufacturer"),
        (4, "Quantum computing advances threaten current encryption standards"),
        (4, "Autonomous weapons system hacked in live demonstration exercise"),
        (4, "Foreign espionage app discovered on government employee devices"),
        (4, "Nuclear plant control systems probed by sophisticated threat actor"),
        (4, "Election infrastructure targeted in coordinated influence operation"),
        (4, "IoT devices weaponized in botnet targeting military installations"),
        (4, "Cybersecurity firm discovers state-sponsored keylogger on classified network"),
        (4, "Biometric data stolen from border control database in major breach"),
    ]

    rows = [{"label": lbl, "text": txt, "threat_category": LABEL_MAP[lbl]}
            for lbl, txt in samples]
    df = pd.DataFrame(rows)
    print(f"[INFO] Built-in dataset: {len(df)} labeled threat intelligence samples\n")
    return df


# ─── 2. Data Cleaning ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Standard NLP preprocessing pipeline:
    lowercase → strip HTML → remove special chars → collapse whitespace
    """
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)           # remove HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)    # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # keep letters/numbers only
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("[STEP 1] Cleaning and preprocessing text data...")
    df = df.dropna(subset=["text", "threat_category"]).copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)

    print(f"         Articles after cleaning: {len(df):,}")
    print(f"         Threat category distribution:")
    counts = df["threat_category"].value_counts()
    for cat, count in counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"           {cat:<35} {count:>5} ({pct:.1f}%) {bar}")
    print()
    return df


# ─── 3. Model Training & Evaluation ──────────────────────────────────────────

def build_models() -> dict:
    """
    Four different classifiers to compare.
    Each is wrapped in a Pipeline with TF-IDF vectorization.
    This is standard production practice — the pipeline ensures
    the same preprocessing applies at train and inference time.
    """
    tfidf_params = dict(
        max_features=50_000,
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,       # log normalization
        min_df=1,
        strip_accents="unicode",
        analyzer="word",
    )

    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs")),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LinearSVC(C=1.0, max_iter=2000)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(**{**tfidf_params, "max_features": 10_000})),
            ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                           random_state=42)),
        ]),
    }


def train_and_evaluate(df: pd.DataFrame) -> tuple:
    print("[STEP 2] Splitting data into train/test sets (80/20)...")

    X = df["clean_text"]
    y = df["threat_category"]

    # Stratified split preserves class balance in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"         Training samples : {len(X_train):,}")
    print(f"         Test samples     : {len(X_test):,}\n")

    models = build_models()
    results = {}

    print("[STEP 3] Training and evaluating models...\n")
    for name, pipeline in models.items():
        print(f"  ▶ {name}")
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"    Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
        print(f"    Train time  : {train_time:.2f}s")
        print()

        results[name] = {
            "pipeline": pipeline,
            "accuracy": acc,
            "y_pred": y_pred,
            "y_test": y_test,
            "train_time": train_time,
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

    # Pick the best model by accuracy
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"  ✓ Best model: {best_name} ({results[best_name]['accuracy']*100:.2f}% accuracy)\n")

    # Full classification report for best model
    print(f"[STEP 4] Detailed report — {best_name}:")
    print(classification_report(
        results[best_name]["y_test"],
        results[best_name]["y_pred"],
        target_names=sorted(LABEL_MAP.values())
    ))

    return results, best_name, X_test, y_test


# ─── 4. Visualization ─────────────────────────────────────────────────────────

def plot_results(results: dict, best_name: str, df: pd.DataFrame):
    print("[STEP 5] Generating visualizations...")

    # ── Figure 1: Model Comparison + Confusion Matrix ──────────────────────────
    fig = plt.figure(figsize=(18, 10), facecolor="#0a0a0a")
    fig.suptitle("THREAT INTELLIGENCE CLASSIFIER — PERFORMANCE REPORT",
                 color="#00ff88", fontsize=16, fontweight="bold",
                 fontfamily="monospace", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot 1: Accuracy comparison bar chart ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    names = list(results.keys())
    accs  = [results[n]["accuracy"] * 100 for n in names]
    short_names = [n.replace(" ", "\n") for n in names]
    bars = ax1.bar(short_names, accs, color=["#27ae60","#3498db","#e74c3c","#f39c12"],
                   edgecolor="#333", linewidth=0.8, zorder=3)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Accuracy (%)", color="#aaaaaa", fontsize=9)
    ax1.set_title("Model Accuracy Comparison", color="#00ff88",
                  fontsize=10, fontweight="bold")
    ax1.set_facecolor("#111111")
    ax1.tick_params(colors="#aaaaaa", labelsize=8)
    ax1.spines[:].set_color("#333333")
    ax1.yaxis.grid(True, color="#222222", zorder=0)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{acc:.1f}%", ha="center", va="bottom",
                 color="white", fontsize=8, fontweight="bold")

    # ── Plot 2: Training time comparison ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    times = [results[n]["train_time"] for n in names]
    bars2 = ax2.barh(short_names, times,
                     color=["#27ae60","#3498db","#e74c3c","#f39c12"],
                     edgecolor="#333", linewidth=0.8)
    ax2.set_xlabel("Train Time (seconds)", color="#aaaaaa", fontsize=9)
    ax2.set_title("Training Speed", color="#00ff88", fontsize=10, fontweight="bold")
    ax2.set_facecolor("#111111")
    ax2.tick_params(colors="#aaaaaa", labelsize=8)
    ax2.spines[:].set_color("#333333")
    ax2.xaxis.grid(True, color="#222222")
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{t:.2f}s", va="center", color="white", fontsize=8)

    # ── Plot 3: Class distribution ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    counts = df["threat_category"].value_counts()
    short_labels = [c.split("/")[0].strip() for c in counts.index]
    wedges, texts, autotexts = ax3.pie(
        counts.values,
        labels=short_labels,
        colors=PALETTE,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"color": "#cccccc", "fontsize": 8},
        wedgeprops={"edgecolor": "#0a0a0a", "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(8)
    ax3.set_title("Threat Category Distribution", color="#00ff88",
                  fontsize=10, fontweight="bold")
    ax3.set_facecolor("#0a0a0a")

    # ── Plot 4: Confusion matrix for best model ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    best = results[best_name]
    labels = sorted(LABEL_MAP.values())
    cm = confusion_matrix(best["y_test"], best["y_pred"], labels=labels)
    short_labels_cm = [l.split("/")[0].strip() for l in labels]

    cmap = LinearSegmentedColormap.from_list("threat", ["#0a0a0a", "#00ff88"])
    im = ax4.imshow(cm, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax4, shrink=0.8).ax.tick_params(colors="#aaaaaa")
    ax4.set_xticks(range(len(labels)))
    ax4.set_yticks(range(len(labels)))
    ax4.set_xticklabels(short_labels_cm, color="#aaaaaa", fontsize=8, rotation=15)
    ax4.set_yticklabels(short_labels_cm, color="#aaaaaa", fontsize=8)
    ax4.set_xlabel("Predicted Label", color="#aaaaaa", fontsize=9)
    ax4.set_ylabel("True Label", color="#aaaaaa", fontsize=9)
    ax4.set_title(f"Confusion Matrix — {best_name}", color="#00ff88",
                  fontsize=10, fontweight="bold")
    ax4.set_facecolor("#111111")
    ax4.spines[:].set_color("#333333")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white", fontweight="bold", fontsize=11)

    # ── Plot 5: F1 score per class for best model ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    report = best["report"]
    cats = [c for c in report if c not in ("accuracy","macro avg","weighted avg")]
    f1s  = [report[c]["f1-score"] for c in cats]
    short_cats = [c.split("/")[0].strip() for c in cats]
    colors_f1  = [COLORS.get(c, "#888888") for c in cats]
    bars5 = ax5.barh(short_cats, f1s, color=colors_f1, edgecolor="#333", linewidth=0.8)
    ax5.set_xlim(0, 1.15)
    ax5.set_xlabel("F1 Score", color="#aaaaaa", fontsize=9)
    ax5.set_title(f"F1 Score by Threat Category\n({best_name})",
                  color="#00ff88", fontsize=10, fontweight="bold")
    ax5.set_facecolor("#111111")
    ax5.tick_params(colors="#aaaaaa", labelsize=8)
    ax5.spines[:].set_color("#333333")
    ax5.xaxis.grid(True, color="#222222")
    for bar, f1 in zip(bars5, f1s):
        ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{f1:.2f}", va="center", color="white", fontsize=9, fontweight="bold")

    path = os.path.join(OUTPUT_DIR, "threat_classifier_report.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"         Saved → {path}")

    # ── Figure 2: Top TF-IDF keywords per category ─────────────────────────────
    best_pipeline = best["pipeline"]
    tfidf = best_pipeline.named_steps["tfidf"]
    clf   = best_pipeline.named_steps["clf"]

    # Only logistic regression and LinearSVC have interpretable coefficients
    if hasattr(clf, "coef_"):
        fig2, axes = plt.subplots(1, 4, figsize=(20, 6), facecolor="#0a0a0a")
        fig2.suptitle("TOP KEYWORDS BY THREAT CATEGORY (TF-IDF Weights)",
                      color="#00ff88", fontsize=14, fontweight="bold",
                      fontfamily="monospace")

        feature_names = np.array(tfidf.get_feature_names_out())
        classes = clf.classes_ if hasattr(clf, "classes_") else sorted(LABEL_MAP.values())

        for ax, cls in zip(axes, classes):
            if hasattr(clf, "classes_"):
                class_idx = list(clf.classes_).index(cls)
            else:
                class_idx = 0

            coefs = clf.coef_[class_idx] if clf.coef_.ndim > 1 else clf.coef_[0]
            top_indices = np.argsort(coefs)[-15:]
            top_words   = feature_names[top_indices]
            top_weights = coefs[top_indices]

            color = COLORS.get(cls, "#27ae60")
            ax.barh(top_words, top_weights, color=color, alpha=0.85, edgecolor="#333")
            ax.set_title(cls.replace("/", "/\n"), color=color,
                         fontsize=9, fontweight="bold")
            ax.set_facecolor("#111111")
            ax.tick_params(colors="#cccccc", labelsize=7)
            ax.spines[:].set_color("#333333")
            ax.xaxis.grid(True, color="#1a1a1a")

        path2 = os.path.join(OUTPUT_DIR, "threat_keywords.png")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
        plt.close()
        print(f"         Saved → {path2}")

    print()


# ─── 5. Interactive Demo ──────────────────────────────────────────────────────

def run_demo(pipeline):
    """
    Let the analyst type any headline or report excerpt
    and instantly see the threat classification + confidence.
    """
    categories = sorted(LABEL_MAP.values())

    print("\n" + "═"*62)
    print("  THREAT INTELLIGENCE CLASSIFIER — LIVE DEMO")
    print("  Type any news headline or report excerpt to classify.")
    print("  Type 'quit' to exit.")
    print("═"*62 + "\n")

    test_examples = [
        "Hackers linked to foreign government breach Pentagon email systems",
        "Protesters storm government building demanding resignation of president",
        "Rare earth mineral export ban threatens defense manufacturing supply",
        "Ballistic missile test conducted over disputed waters raises alarms",
    ]
    print("  Example headlines (auto-classifying to show you how it works):\n")
    for ex in test_examples:
        pred = pipeline.predict([ex])[0]
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            proba = pipeline.predict_proba([ex])[0]
            conf  = max(proba) * 100
            conf_str = f"  Confidence: {conf:.1f}%"
        else:
            # LinearSVC uses decision function distance as proxy
            scores = pipeline.decision_function([ex])[0]
            conf_str = ""

        color_codes = {
            "Geopolitical / Conflict":   "\033[91m",   # red
            "Civil Unrest / Operations": "\033[93m",   # yellow
            "Economic Threat":           "\033[33m",   # dark yellow
            "Cyber / Technology Threat": "\033[92m",   # green
        }
        reset = "\033[0m"
        c = color_codes.get(pred, "")
        print(f"  Headline : {ex}")
        print(f"  Category : {c}[ {pred} ]{reset}{conf_str}")
        print()

    print("─"*62)
    print("  Now try your own:\n")

    while True:
        try:
            user_input = input("  Enter headline (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q", ""):
            print("\n  Exiting demo. Stay vigilant.\n")
            break

        pred = pipeline.predict([user_input])[0]
        c = color_codes.get(pred, "")
        print(f"\n  ▶ Classification: {c}[ {pred} ]{reset}\n")


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    demo_only = "--demo" in sys.argv

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   THREAT INTELLIGENCE NEWS CLASSIFIER                   ║")
    print("║   Homeland Security & Defense NLP Pipeline              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Load data
    df = load_data()

    if demo_only:
        # Quick train on all data for demo mode
        print("[INFO] Training model for demo (no evaluation)...")
        df = preprocess(df)
        pipeline = build_models()["Logistic Regression"]
        pipeline.fit(df["clean_text"], df["threat_category"])
        run_demo(pipeline)
        return

    # Full pipeline
    df       = preprocess(df)
    results, best_name, X_test, y_test = train_and_evaluate(df)
    plot_results(results, best_name, df)

    print("═"*62)
    print(f"  Pipeline complete. Outputs saved to: ./{OUTPUT_DIR}/")
    print("═"*62)

    # Always end with the interactive demo
    best_pipeline = results[best_name]["pipeline"]
    run_demo(best_pipeline)


if __name__ == "__main__":
    main()
