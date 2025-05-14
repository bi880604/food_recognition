import os, urllib.request, torch, torchvision.models as models
MODEL_URL = "https://huggingface.co/your-hf-name/food-recognition-space/resolve/main/resnet_food_model.pt"
MODEL_PATH = "resnet_food_model.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()
