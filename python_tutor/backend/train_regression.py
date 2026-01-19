import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("dataset.csv")

X = df[["lines_of_code", "errors_count", "difficulty"]]
y = df["score"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "regression_model.pkl")
print("Regression Model Trained!")
