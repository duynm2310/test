from flask import Flask, request, jsonify
from feature_extractor import Feature_Extraction_img
import joblib
import os
from werkzeug.utils import secure_filename
import tempfile
import numpy as np


app = Flask(__name__)

svm_model = joblib.load('train/model.pkl')

@app.route("/")
def home():
    return "model is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Đang xử lý yêu cầu...")
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        img_file = request.files["image"]
        if img_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        print(f"Đã nhận tệp ảnh: {img_file.filename}")

        # Lưu ảnh vào thư mục tạm thời
        temp_dir = "./temp"  # Thư mục tạm
        img_filename = secure_filename(img_file.filename)
        temp_file_path = os.path.join(temp_dir, img_filename)  # Lưu ảnh tại temp_dir

        img_file.save(temp_file_path)
        print(f"Đã lưu ảnh tạm thời tại: {temp_file_path}")

        # Feature extraction
        features = Feature_Extraction_img(temp_file_path)
        print(f"Đặc trưng đã trích xuất: {features}")
        # Predict
        prediction = svm_model.predict(features)
        print(f"Dự đoán: {prediction}")

        # Xóa tệp tạm
        os.remove(temp_file_path)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        print("🔥 Error during prediction:", str(e))  # In ra lỗi chi tiết
        return jsonify({"error": "Internal Server Error"}), 500
    
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)