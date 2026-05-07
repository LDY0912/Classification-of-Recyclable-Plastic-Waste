# 외부 하드로부터 json경로를 추출

import os

# 정확한 시작 루트 경로 (공백 포함 문자열 처리)
root_dir = "/Volumes/My Passport/Recyclables_Sorting_Data/Official_Open_Data/Validation/Label_Data"

# 모든 하위 디렉토리에서 .json 파일 경로 수집 (Colored 폴더 제외)
json_paths = []
for subdir, _, files in os.walk(root_dir):
    if "Colored" in subdir:
        continue  # 'Colored' 포함된 폴더는 무시
    for file in files:
        if file.endswith(".json"):
            json_paths.append(os.path.join(subdir, file))

# 저장할 경로
output_path = "data/json.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 경로 저장
with open(output_path, "w", encoding="utf-8") as f:
    for path in json_paths:
        f.write(path + "\n")
