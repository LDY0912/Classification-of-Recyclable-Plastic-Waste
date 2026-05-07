import sys
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report, confusion_matrix
import os

# 파라미터 설정
MODEL_PATH = "outputs/efficientnet-b4-6ed6700e.pth"
DATA_PATH = "data/pollution/test.txt"
BATCH_SIZE = 32
CLASS_NAMES = ["clean", "polluted"]

# 데이터셋 클래스
class PollutionTestDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.samples = []
        self.transform = transform
        with open(txt_path, "r") as f:
            for line in f:
                path, label = line.strip().split(",")
                label_id = 0 if label == "clean" else 1
                self.samples.append((path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터 로딩
dataset = PollutionTestDataset(DATA_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_name("efficientnet-b4")
model._fc = torch.nn.Linear(model._fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 예측
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels)

# 리포트 출력
print(f"\n📁 결과 - {os.path.basename(DATA_PATH)}")
print("▶ Accuracy:", (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item())
print("\n▶ Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
print("▶ Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
