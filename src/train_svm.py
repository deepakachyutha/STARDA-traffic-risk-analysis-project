import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score

print("🐢 Training SVM (Experimental Benchmark)...")

data_path = "data/modern_training_data.csv"
if not os.path.exists(data_path): data_path = "notebook/modern_training_data.csv"
df = pd.read_csv(data_path)

# SVM CRASH PREVENTION: Limit to 10k rows
print("⚠️ Sampling 10,000 rows for SVM stability...")
df = df.sample(10000, random_state=42)

X = df.drop(columns=['Severity'], errors='ignore')
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(probability=True, kernel='rbf', random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')

print(f"SVM Accuracy: {acc:.4f} | F1: {f1:.4f}")

joblib.dump(model, "models/svm_model.pkl")

with open("research_files/report_SVM.txt", "w") as f:
    f.write(classification_report(y_test, preds, zero_division=0))

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap='Purples', values_format='d')
plt.title(f"SVM (Limited Data)\nAccuracy: {acc:.2%}")
plt.savefig("research_files/confusion_matrix_SVM.png")
print("Done.")