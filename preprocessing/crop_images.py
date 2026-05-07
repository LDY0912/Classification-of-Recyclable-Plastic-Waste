import os
from PIL import Image

# 경로 설정
image_dir = "data/original/images"
label_dir = "data/original/labels"
output_dir = "data/cropped_images"

# 출력 폴더 생성
os.makedirs(output_dir, exist_ok=True)

# 이미지 목록 순회
for filename in os.listdir(image_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

    # 라벨 파일이 없으면 스킵
    if not os.path.exists(label_path):
        print(f"라벨 없음: {label_path}")
        continue

    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # 라벨 읽기
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # YOLO 포맷이 아니면 스킵

        class_id, x, y, bw, bh = parts
        x, y, bw, bh = map(float, (x, y, bw, bh))

        # 바운딩박스 좌표 계산
        left = int((x - bw / 2) * w)
        top = int((y - bh / 2) * h)
        right = int((x + bw / 2) * w)
        bottom = int((y + bh / 2) * h)

        cropped = image.crop((left, top, right, bottom))

        # 저장 파일명 구성 (중복 방지 위해 인덱스 추가)
        base = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base}_{idx}.jpg")
        cropped.save(output_path)
