import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- CONFIGURATION ---
NUM_SIMULATIONS = 5 
DATA_PATH = "data/modern_training_data.csv"
if not os.path.exists(DATA_PATH): DATA_PATH = "notebook/modern_training_data.csv"

print(f"Starting Monte Carlo Simulation ({NUM_SIMULATIONS} Iterations)...")
print("   This measures Model Stability and Confidence Intervals.")

# 1. Load Data
df = pd.read_csv(DATA_PATH)
if len(df) > 100000: 
    print("  Downsampling to 100k rows for simulation speed...")
    df = df.sample(100000, random_state=42)

X = df.drop(columns=['Severity'], errors='ignore')
y = df['Severity']

results = {
    "Random Forest": {"acc": [], "f1": []},
    "AdaBoost":      {"acc": [], "f1": []},
    "SVM":           {"acc": [], "f1": []}
}


start_time = time.time()

for i in range(NUM_SIMULATIONS):
    seed = 42 + i 
    print(f"\n🔄 Simulation {i+1}/{NUM_SIMULATIONS} (Seed {seed})...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results["Random Forest"]["acc"].append(accuracy_score(y_test, rf_preds))
    results["Random Forest"]["f1"].append(f1_score(y_test, rf_preds, average='weighted'))
    
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, random_state=seed)
    ada.fit(X_train, y_train)
    ada_preds = ada.predict(X_test)
    results["AdaBoost"]["acc"].append(accuracy_score(y_test, ada_preds))
    results["AdaBoost"]["f1"].append(f1_score(y_test, ada_preds, average='weighted'))
    
    svm = SVC(probability=False, kernel='rbf', random_state=seed) 
    svm.fit(X_train[:10000], y_train[:10000])
    svm_preds = svm.predict(X_test) 
    results["SVM"]["acc"].append(accuracy_score(y_test, svm_preds))
    results["SVM"]["f1"].append(f1_score(y_test, svm_preds, average='weighted'))

print("\n" + "="*50)
print(f"FINAL RESULTS ({NUM_SIMULATIONS} Simulations)")
print("="*50)
print(f"{'Model':<15} | {'Mean Accuracy':<15} | {'Std Dev':<10} | {'Mean F1':<10}")
print("-" * 55)

summary_text = "simulation_report.txt"
with open(summary_text, "w") as f:
    f.write(f"Simulation Report ({NUM_SIMULATIONS} Runs)\n\n")
    
    for name, metrics in results.items():
        mean_acc = np.mean(metrics["acc"])
        std_acc = np.std(metrics["acc"])
        mean_f1 = np.mean(metrics["f1"])
        
        print(f"{name:<15} | {mean_acc:.4f}          | ±{std_acc:.4f}   | {mean_f1:.4f}")
        
        f.write(f"Model: {name}\n")
        f.write(f"   Mean Accuracy: {mean_acc:.4f} (±{std_acc:.4f})\n")
        f.write(f"   Mean F1 Score: {mean_f1:.4f}\n\n")

print("-" * 55)
print(f"Report saved to {summary_text}")
print(f"Total Time: {time.time() - start_time:.1f}s")