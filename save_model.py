import torch
import torchvision.models as models

# 載入預訓練模型
model = models.resnet18(pretrained=True)

# 儲存模型
torch.save(model.state_dict(), "resnet_food_model.pt")
