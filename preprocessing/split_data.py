import json
import random
from pathlib import Path

CROPPED_DIR = Path("data/cropped_images")
JSON_LIST_PATH = Path("data/json_filtered.txt")

INTACT_KEYWORD = "원형"
CLEAN_KEYWORD = "오염없음"

json_paths = JSON_LIST_PATH.read_text(encoding="utf-8").splitlines()

# { 파일명_without_ext: json_path }
json_map = {
    Path(j).stem: j
    for j in json_paths
}

damage_list = []
pollution_list = []

for img_path in CROPPED_DIR.glob("*.jpg"):
    stem_full = img_path.stem                      # e.g. "878233@0_04002_220907_P1_T3__1164_0"
    base_stem = stem_full.rsplit("_", 1)[0]        # e.g. "878233@0_04002_220907_P1_T3__1164"

    json_path = json_map.get(base_stem)
    if not json_path:
        print(f"JSON 없음: {img_path.name}")
        continue

    try:
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)
        ann_list = meta.get("ANNOTATION_INFO", [])
        if not ann_list:
            print(f"Empty annotation: {json_path}")
            continue

        ann = ann_list[0]
        damage_type = ann.get("DAMAGE", "")
        dirtiness = ann.get("DIRTINESS", "")

        damage_class = "intact" if damage_type == INTACT_KEYWORD else "damage"
        pollution_class = "clean" if dirtiness == CLEAN_KEYWORD else "polluted"

        damage_list.append((str(img_path), damage_class))
        pollution_list.append((str(img_path), pollution_class))

    except Exception as e:
        print(f"오류 발생: {json_path} / {e}")

def split_and_save(data, category):
    random.shuffle(data)
    n = len(data)
    split = {
        "train": data[:int(n*0.6)],
        "val": data[int(n*0.6):int(n*0.8)],
        "test": data[int(n*0.8):]
    }

    out_dir = Path(f"data/{category}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in split.items():
        with open(out_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
            for path, cls in items:
                f.write(f"{path},{cls}\n")

split_and_save(damage_list, "damage")
split_and_save(pollution_list, "pollution")

