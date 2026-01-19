import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("dataset.csv")

X = df[["score", "mistakes"]]
y = df["pass"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "rf_model.pkl")
print("Random Forest Model Trained!")
