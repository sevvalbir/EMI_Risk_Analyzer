"""Train RandomForest EMI classifier and persist model artifact."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FEATURE_COLUMNS
from src.paths import DATASET_PATH, MODEL_PATH, MODELS_DIR, ensure_dir

df = pd.read_csv(DATASET_PATH)
X = df[FEATURE_COLUMNS]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

importance = (
    pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    )
    .sort_values(by="Importance", ascending=False)
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10, 6))
plt.barh(importance["Feature"], importance["Importance"])
plt.xlabel("Importance")
plt.ylabel("PCB Parameters")
plt.title("EMI Risk Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

ensure_dir(MODELS_DIR)
joblib.dump(model, MODEL_PATH)
print("Model saved:", MODEL_PATH.resolve())

plt.show()
