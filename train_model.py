import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("student_constraints.csv")

features = [
    "device_tier",
    "daily_power_hours",
    "daily_data_mb",
    "campus_lab_access",
    "peer_group_score",
    "weekly_practice_hours",
]

X = df[features]
y = df["target_phase"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred).tolist()
report = classification_report(y_test, y_pred, output_dict=True)
feature_importances = dict(zip(features, model.feature_importances_.round(4).tolist()))

print(f"Model Test Accuracy: {acc:.2%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")

metrics_payload = {
    "accuracy": round(acc, 4),
    "confusion_matrix": cm,
    "feature_importances": feature_importances,
    "classification_report": report,
    "test_samples": len(y_test),
    "train_samples": len(y_train),
}

with open("metrics.json", "w") as f:
    json.dump(metrics_payload, f, indent=4)

print("Saved model.pkl and metrics.json successfully.")
