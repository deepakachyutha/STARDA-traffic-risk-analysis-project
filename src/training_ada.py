import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score

print("Training AdaBoost (Accuracy-Optimized)...")

data_path = "data/modern_training_data.csv"
if not os.path.exists(data_path): data_path = "notebook/modern_training_data.csv"
df = pd.read_csv(data_path)

if len(df) > 50000: df = df.sample(50000, random_state=42)

X = df.drop(columns=['Severity'], errors='ignore')
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard AdaBoost
model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')

print(f"AdaBoost Accuracy: {acc:.4f} | F1: {f1:.4f}")

joblib.dump(model, "models/adaboost_model.pkl")

with open("research_files/report_AdaBoost.txt", "w") as f:
    f.write(classification_report(y_test, preds, zero_division=0))

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap='Oranges', values_format='d')
plt.title(f"AdaBoost\nAccuracy: {acc:.2%}")
plt.savefig("research_files/confusion_matrix_AdaBoost.png")
print("Done.")