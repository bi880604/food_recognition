from flask import Flask, request, jsonify
from PIL import Image
import io, torchvision.transforms as T
import model  # 會觸發下載並載入

app = Flask(__name__)
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "need image"}), 400
    img_bytes = request.files["image"].read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model.model(tensor)
        cls = out.argmax().item()
    return jsonify({"class_id": int(cls)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
