import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

# 하이퍼파라미터 및 설정
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
TRAIN_TXT = "data/pollution/train.txt"
VAL_TXT = "data/pollution/val.txt"  
MODEL_PATH = "outputs/efficientnet-b4-6ed6700e.pth"

os.makedirs("outputs", exist_ok=True)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CroppedImageDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.samples = []
        self.transform = transform
        with open(txt_file, 'r') as f:
            for line in f:
                image_path, cls = line.strip().split(',')
                label = 0 if cls in ['clean', 'intact'] else 1
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    train_dataset = CroppedImageDataset(TRAIN_TXT, transform=train_transform)
    val_dataset = CroppedImageDataset(VAL_TXT, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 모델 불러오기
    model = EfficientNet.from_name('efficientnet-b4')

    try:
        state_dict = torch.load('weights/efficientnet-b4-6ed6700e.pth', weights_only=True)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print("사전 학습된 가중치를 찾을 수 없어 기본 구조로 시작합니다.")

    model._fc = nn.Linear(model._fc.in_features, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # 훈련 단계
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 검증 단계
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        # Early Stopping 로직
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"🔥 Best Model Updated! Saving to {MODEL_PATH}...")
            torch.save(model.state_dict(), MODEL_PATH)