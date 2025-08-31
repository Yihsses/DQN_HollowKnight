
from dqn_3cnn import Q_construct_3d
from Tool import framebuffer
from Tool import screngrap
from Tool.CoordinateBuffer import CoordinateClient
import time
from torchvision import models, transforms
import torch.nn as nn
import torch
import torch.nn.functional as F
class_names = ["down", "nomove", "nothing", "rusg", "shot", "skill"]

# 建立模型結構
num_classes = len(class_names)
model = models.resnet18(pretrained=False)  # 這裡不用 pretrained
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("YOLO/resnet18_best.pth", map_location=device))
model.to(device)
model.eval()
model.eval()  # 切換成推理模式
# 載入訓練好的權重

cb =   CoordinateClient()
dx = 45.0892857143
dy = (525-25)/(39-28.6)
# 40 - 28.6
number = 1000
pred_class = "nothing"

while True:
    try:
        number += 1 
        temp = cb.get_coordinates()
        x = (temp[0]['x']-15.3) * dx +50
        y = 525-(temp[0]['y']-28.6) * dy  

        output  = model(screngrap.screngrap.grap(
            'HOLLOW KNIGHT', d_height=180, d_width=200, d_top=y, d_left=x, img2_return=True
        ).to(device))

        # 轉成機率
        probs = F.softmax(output, dim=1)
        max_prob, pred_idx = torch.max(probs, 1)    # 找最大值和索引

        # 取得其他類別機率
        other_probs = probs[0, [i for i in range(len(class_names)) if i != pred_idx.item()]]

        # 條件判斷
        if max_prob.item() > 0.8 and torch.all(other_probs < 0.3):
            pred_class = class_names[pred_idx.item()]
            prev_class = pred_class
        else:
            pred_class = prev_class  # 維持上一個動作

        print(pred_class)
        time.sleep(0.1)

    except Exception as e:
        print("Error:", e)