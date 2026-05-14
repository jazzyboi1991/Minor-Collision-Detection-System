import json
import os


def merge_datasets():
    # 파일 경로 설정
    base_path = os.path.dirname(os.path.abspath(__file__))
    positions_path = os.path.join(base_path, 'reversed_positions.json')
    rgb_values_path = os.path.join(
        base_path, 'reversed_average_RGB_values.json')
    brightness_values_path = os.path.join(
        base_path, 'reversed_average_brightness_values.json')

    # 결과 저장 디렉토리 (Data/Parking Vehicle Hit an Run Dataset/JSON Files)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(base_path))), "JSON Files")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'reversed_dataset.json')

    # 데이터 로드
    print(f"데이터를 불러오는 중...")
    try:
        with open(positions_path, 'r', encoding='utf-8') as f:
            positions_data = json.load(f)

        with open(rgb_values_path, 'r', encoding='utf-8') as f:
            rgb_data = json.load(f)

        with open(brightness_values_path, 'r', encoding='utf-8') as f:
            brightness_data = json.load(f)
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. {e}")
        return
    except json.JSONDecodeError as e:
        print(f"오류: JSON 파일 형식이 올바르지 않습니다. {e}")
        return

    # 데이터 병합
    dataset = {}
    merge_count = 0
    missing_data_count = 0

    print(f"데이터 병합 중...")
    for video_id, pos_info in positions_data.items():
        # 모든 파일에서 확장자를 제외한 ID를 키로 사용하므로 직접 매칭
        # 만약 키 불일치가 걱정된다면 아래와 같이 정규화 로직을 유지할 수 있습니다.
        key = video_id.replace('.mp4', '')

        if key in rgb_data and key in brightness_data:
            # 모든 정보가 있는 경우 병합
            merged_info = pos_info.copy()
            merged_info.update(rgb_data[key])
            merged_info['average_brightness'] = brightness_data[key]

            dataset[video_id] = merged_info
            merge_count += 1
        else:
            missing_data_count += 1

    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"작업 완료!")
    print(f"총 병합된 항목 수: {merge_count}")
    if missing_data_count > 0:
        print(f"데이터 누락 항목 수: {missing_data_count}")
    print(f"결과가 {output_path}에 저장되었습니다.")


if __name__ == "__main__":
    merge_datasets()
