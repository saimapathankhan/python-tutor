import pandas as pd
from sklearn.svm import SVC
import joblib

df = pd.read_csv("dataset.csv")

X = df[["score"]]
y = df["level"]

model = SVC()
model.fit(X, y)

joblib.dump(model, "svm_model.pkl")
print("SVM Model Trained!")
