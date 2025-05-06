from flask import Flask, request, jsonify
from feature_extractor import Feature_Extraction_img
import pickle

app = Flask(__name__)

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

@app.route("/")
def home():
    return "model is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    features = Feature_Extraction_img(image_file).reshape(1, -1)
    prediction = svm_model.predict(features)[0]

    return jsonify({"prediction": int(prediction)})
