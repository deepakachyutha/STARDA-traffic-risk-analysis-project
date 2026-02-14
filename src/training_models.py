import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import os

# --- SETUP ---
print("Loading Data...")
data_path = "data/modern_training_data.csv"
if not os.path.exists(data_path):
    data_path = "notebook/modern_training_data.csv"

# Load Data
df = pd.read_csv(data_path)
print(f"  Dataset loaded: {len(df)} rows")

if len(df) > 50000:
    print("  Dataset > 50k rows. Using 50k sample for fast training.")
    df_sample = df.sample(50000, random_state=42)
else:
    df_sample = df

X = df_sample.drop(columns=['Severity'], errors='ignore')
y = df_sample['Severity']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("research_files", exist_ok=True)

# Define Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, random_state=42),
    
    "SVM": SVC(probability=True, kernel='rbf', random_state=42) 
}

results = {}

print("\nStarting Comprehensive Evaluation...")

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    
    if name == "SVM":
       
        print("SVM detected: Limiting training data to 10,000 rows to prevent freeze...")
        model.fit(X_train[:10000], y_train[:10000])
    else:
        model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
  
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    report = classification_report(y_test, preds, zero_division=0) 
    cm = confusion_matrix(y_test, preds)
    
    results[name] = {"Accuracy": acc, "F1": f1}
    
    filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, filename)
    
    with open(f"research_files/report_{name.replace(' ', '_')}.txt", "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write(report)
        
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{name} Confusion Matrix\nAccuracy: {acc:.2%}")
    plt.savefig(f"research_files/confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()
    
    print(f" Saved Model -> {filename}")
    print(f" Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

print("\n" + "="*30)
print("FINAL LEADERBOARD")
print("="*30)
df_results = pd.DataFrame(results).T.sort_values(by="F1", ascending=False)
print(df_results)
print("="*30)