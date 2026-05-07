import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import ViTForImageClassification

# 설정
TXT_PATH = "data/damage/test.txt"
MODEL_PATH = "outputs/vit_damage.pth"
LABEL2ID = {"intact": 0, "damage": 1}
ID2LABEL = {0: "intact", 1: "damage"}

# Transform
transform = transforms.Compose([
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DamageDataset(TXT_PATH, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(pixel_values=images).logits
            predictions = torch.argmax(outputs, dim=1).cpu().tolist()
            preds.extend(predictions)
            targets.extend(labels.tolist())

    acc = accuracy_score(targets, preds)
    report = classification_report(targets, preds, target_names=LABEL2ID.keys())
    matrix = confusion_matrix(targets, preds)

    print(f"▶ Accuracy: {acc}")
    print("\n▶ Classification Report:\n", report)
    print("▶ Confusion Matrix:\n", matrix)
