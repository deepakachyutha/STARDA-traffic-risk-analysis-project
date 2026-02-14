import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

print("🌲 Training Random Forest (Safety-Optimized)...")

# 1. Load & Sample
data_path = "data/modern_training_data.csv"
if not os.path.exists(data_path): data_path = "notebook/modern_training_data.csv"
df = pd.read_csv(data_path)

if len(df) > 100000: df = df.sample(100000, random_state=42)

X = df.drop(columns=['Severity'], errors='ignore')
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')

print(f"Random Forest Accuracy: {acc:.4f} | F1: {f1:.4f}")

os.makedirs("models", exist_ok=True)
os.makedirs("research_files", exist_ok=True)

joblib.dump(model, "models/random_forest_model.pkl")

with open("research_files/report_Random_Forest.txt", "w") as f:
    f.write(classification_report(y_test, preds))

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap='Greens', values_format='d')
plt.title(f"Random Forest (Balanced)\nAccuracy: {acc:.2%}")
plt.savefig("research_files/confusion_matrix_Random_Forest.png")
print("Done.")