import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification

# 하이퍼파라미터 및 설정
TRAIN_TXT = "data/damage/train.txt"
VAL_TXT = "data/damage/val.txt" 
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-5
MODEL_PATH = "outputs/vit_damage.pth"
LABEL2ID = {"intact": 0, "damage": 1}
ID2LABEL = {0: "intact", 1: "damage"}

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

class DamageDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.samples = []
        self.transform = transform
        with open(txt_path, "r") as f:
            for line in f:
                img_path, label = line.strip().split(",")
                self.samples.append((img_path, LABEL2ID[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    train_dataset = DamageDataset(TRAIN_TXT, train_transform)
    val_dataset = DamageDataset(VAL_TXT, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0 

    for epoch in range(EPOCHS):
        # 훈련 단계
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
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
                outputs = model(pixel_values=images).logits
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        # Early Stopping 로직
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"🔥 Best Model Updated! Saving to {MODEL_PATH}...")
            torch.save(model.state_dict(), MODEL_PATH)