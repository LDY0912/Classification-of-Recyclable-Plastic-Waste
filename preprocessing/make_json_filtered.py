# 모든 사진을 다운받진 못함 그거에따라 json도 같은 이름만 남기고 나머지는 삭제하는 코드

from pathlib import Path

# 파일 경로
json_txt_path = Path("data/json.txt")
image_dir = Path("data/original/images")
output_path = Path("data/json_filtered.txt")

# 원본 json 목록 읽기
all_json_paths = json_txt_path.read_text(encoding="utf-8").splitlines()

# 이미지 폴더 기준으로 존재하는 파일명 추출 (확장자 없이)
existing_images = {p.stem for p in image_dir.glob("*.jpg")}

# 필터링된 json 목록 생성
filtered_jsons = [jp for jp in all_json_paths if Path(jp).stem in existing_images]

# 결과 저장
with open(output_path, "w", encoding="utf-8") as f:
    for path in filtered_jsons:
        f.write(path + "\n")

print(f"필터링 완료: {len(filtered_jsons)}개 JSON 경로 저장됨 → {output_path}")
