import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# Load your dataset
df = pd.read_csv("full_crop_data.csv")

# Features and target
X = df.drop("Crop", axis=1)
y = df["Crop"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
print(f"📌 Precision: {precision:.2f}")
print(f"📌 Recall: {recall:.2f}")
print(f"📌 F1-Score: {f1:.2f}")

print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
plt.figure(figsize=(14, 10))
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ✅ Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n🎉 Model trained and saved successfully as 'model.pkl'")
