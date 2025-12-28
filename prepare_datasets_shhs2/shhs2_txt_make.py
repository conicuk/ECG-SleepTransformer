import os
import random

# 설정
annotation_folder = r"E:\coding\sleep_stage\data\shhs\polysomnography\annotations-events-profusion\shhs2"
save_txt_path = r"E:\coding\sleep_stage\prepare_datasets_shhs2\selected_shhs2_files.txt"
sample_count = 100  # 선택할 피험자 수

def has_faulty_label(xml_path):
    """XML 파일에 잘못된 라벨이 있는지 확인 (라벨이 5보다 크면 faulty)"""
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for stage in root[4]:  # 수면 단계 정보
            lbl = int(stage.text)
            if lbl > 5:
                return True
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {xml_path}, 무시됨. ({e})")
        return True  # 오류 있는 파일은 faulty로 간주

all_files = os.listdir(annotation_folder)

mesa_ids = []
for filename in all_files:
    if filename.startswith("shhs2-") and "-profusion" in filename:
        base_name = filename.split("-profusion")[0]
        mesa_ids.append(base_name)

mesa_ids = sorted(set(mesa_ids), key=lambda x: int(x.split("-")[-1]))

valid_ids = []
for id_ in mesa_ids:
    xml_path = os.path.join(annotation_folder, f"{id_}-profusion.xml")
    if not has_faulty_label(xml_path):
        valid_ids.append(id_)

print(f"⚙️ 유효한 ID 수: {len(valid_ids)}개")

# 무작위 100개 샘플링
random.seed(42)
selected_ids = random.sample(valid_ids, sample_count)
selected_ids.sort(key=lambda x: int(x.split("-")[-1]))

with open(save_txt_path, "w") as f:
    for id_ in selected_ids:
        f.write(id_ + "\n")

print(f"✅ {len(selected_ids)}개 ID가 저장되었습니다: {save_txt_path}")

