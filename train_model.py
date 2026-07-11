import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Dataset Load
data = pd.read_csv("Diabetes_prediction.csv")

# separating inputs and outputs
X = data.drop("Diagnosis", axis=1)
y = data["Diagnosis"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# make a model
model = RandomForestClassifier(random_state=42)

# train a model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

# make a model folder
os.makedirs("models", exist_ok=True)

# Model Save
joblib.dump(model, "models/diabetes_model.pkl")

print("Model Successfully Saved!")