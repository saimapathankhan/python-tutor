import joblib
import numpy as np

# load PKL models
regression_model = joblib.load("regression_model.pkl")
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")

def predict_regression(value):
    value = np.array([[value]], dtype=float)
    return float(regression_model.predict(value)[0])

def predict_svm(value):
    value = np.array([[value]], dtype=float)
    return int(svm_model.predict(value)[0])

def predict_rf(value):
    value = np.array([[value]], dtype=float)
    return int(rf_model.predict(value)[0])
