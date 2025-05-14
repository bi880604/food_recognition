# app.py
from flask import Flask, request, jsonify
from model import predict_image
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return "AI 食物辨識 API 正常運作中"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "請上傳圖片"}), 400
    
    image_file = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    label, confidence = predict_image(image_path)
    return jsonify({"prediction": label, "confidence": f"{confidence}%"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
