from flask import Flask, request, jsonify
from ml_functions import predict_regression, predict_svm, predict_rf

app = Flask(__name__)

@app.route("/predict/regression", methods=["POST"])
def api_regression():
    value = float(request.json["value"])
    return jsonify({"prediction": predict_regression(value)})

@app.route("/predict/svm", methods=["POST"])
def api_svm():
    value = float(request.json["value"])
    return jsonify({"prediction": predict_svm(value)})

@app.route("/predict/rf", methods=["POST"])
def api_rf():
    value = float(request.json["value"])
    return jsonify({"prediction": predict_rf(value)})

if __name__ == "__main__":
    app.run(debug=True)
