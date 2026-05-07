import os
import torch
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from transformers import ViTForImageClassification

# 설정 및 모델 준비
image_dir = "data/original/images"
label_dir = "data/original/labels"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 오염도 판별 모델 (EfficientNet-B4) 로드
eff_model = EfficientNet.from_name("efficientnet-b4")
eff_model._fc = torch.nn.Linear(eff_model._fc.in_features, 2)
eff_model.load_state_dict(torch.load("outputs/efficientnet-b4-6ed6700e.pth", map_location=device))
eff_model.to(device)
eff_model.eval()

# 훼손도 판별 모델 (ViT) 로드
vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", 
    num_labels=2, 
    ignore_mismatched_sizes=True
)
vit_model.load_state_dict(torch.load("outputs/vit_damage.pth", map_location=device))
vit_model.to(device)
vit_model.eval()

# 이미지 전처리 (두 모델 공통 사용 가능)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 메모리 상에서 직접 처리하는 파이프라인
print("I/O 최적화 파이프라인 추론 시작...")

with torch.no_grad(): # 추론 과정이므로 그래디언트 계산 비활성화
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        # 1. 원본 이미지 1회 로드
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5: continue

            class_id, x, y, bw, bh = map(float, parts)

            # 2. 바운딩 박스 좌표 계산
            left = int((x - bw / 2) * w)
            top = int((y - bh / 2) * h)
            right = int((x + bw / 2) * w)
            bottom = int((y + bh / 2) * h)

            # 3. 메모리 상에서 이미지 자르기
            cropped_image = image.crop((left, top, right, bottom))

            # 4. 자른 이미지를 곧바로 Tensor로 변환하여 모델 입력 형태로 준비
            # unsqueeze(0)를 통해 배치 차원(batch size=1)을 추가
            input_tensor = transform(cropped_image).unsqueeze(0).to(device)

            # 5. 병렬 모델 추론 (로컬 메모리의 텐서를 그대로 활용)
            # EfficientNet (오염도)
            pollution_outputs = eff_model(input_tensor)
            pollution_pred = torch.argmax(pollution_outputs, dim=1).item()
            pollution_label = "polluted" if pollution_pred == 1 else "clean"

            # ViT (훼손도)
            damage_outputs = vit_model(pixel_values=input_tensor).logits
            damage_pred = torch.argmax(damage_outputs, dim=1).item()
            damage_label = "damage" if damage_pred == 1 else "intact"

            print(f"[{filename} - 객체 {idx}] 상태: {pollution_label}, {damage_label}")