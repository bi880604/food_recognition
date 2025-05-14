# model.py
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image

# 載入模型
model = resnet50(pretrained=True)
model.eval()

# 載入 imagenet 分類名稱
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# 圖片前處理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        out = model(batch_t)
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return labels[index[0]], round(percentage[index[0]].item(), 2)
