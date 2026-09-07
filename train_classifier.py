"""
Train Classifier for Isolated Sign Recognition
------------------------------------------------------------
Reads features.csv (from extract_features.py) and trains a RandomForest
classifier. This trains in seconds even on a small dataset and is much
more reliable than a deep model when data/time is limited.

Saves: sign_classifier.pkl (model + label list)

RUN:
    pip install -r requirements.txt
    python train_classifier.py
"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

FEATURES_CSV = "features.csv"
MODEL_OUT = "sign_classifier.pkl"


def main():
    df = pd.read_csv(FEATURES_CSV)

    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].astype(str).to_numpy(dtype=object)

    print(f"Loaded {len(df)} samples across {df['label'].nunique()} signs: {sorted(df['label'].unique())}")

    if len(df) < 10:
        print("⚠️ Very few samples — record more takes per sign if accuracy is low.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if df["label"].nunique() > 1 else None
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n✅ Test accuracy: {acc:.2%}")
    print(classification_report(y_test, preds, zero_division=0))

    # Retrain on ALL data for the final model used in the live demo
    final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(X, y)

    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": feature_cols}, f)

    print(f"Saved final model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
