import pandas as pd
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

# 1. LOAD DATA
print("Loading modern_training_data.csv...")
df = pd.read_csv("/Users/deepakachyutha/deepak/car crash research project/notebook/modern_training_data.csv")

X = df.drop(columns=['Severity'])
y = df['Severity']

print(f"Data Loaded. Shape: {df.shape}")

# 2. SPLIT TRAINING & TESTING
print("Splitting data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN MODEL
print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Training Complete!")

# --- 4. EVALUATE ---
print("Evaluating Performance...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nFINAL TEST ACCURACY: {acc:.2%}")

print("\n--- DETAILED CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))


joblib.dump(model, "accident_severity_model.pkl")
print("Model saved as 'accident_severity_model.pkl' (Ready for Chatbot)")
