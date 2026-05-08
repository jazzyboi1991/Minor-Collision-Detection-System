import os
import json


def convert_dataset_to_json():
    # 현재 스크립트 파일의 경로를 기준으로 프로젝트 루트 및 데이터셋 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 데이터셋 위치를 부모 디렉토리(Normal)로 설정
    dataset_dir = os.path.join(os.path.dirname(base_dir), "txt")
    output_file = os.path.join(base_dir, "positions.json")

    # 결과 데이터를 담을 딕셔너리
    combined_data = {}

    # Dataset 폴더 내의 모든 .txt 파일 목록 가져오기
    if not os.path.exists(dataset_dir):
        print(f"오류: '{dataset_dir}' 폴더를 찾을 수 없습니다.")
        return

    txt_files = [f for f in os.listdir(dataset_dir) if f.endswith('.txt')]
    print(f"총 {len(txt_files)}개의 어노테이션 파일을 처리 중...")

    for filename in sorted(txt_files):
        file_id = os.path.splitext(filename)[0]  # 파일명(확장자 제외)을 키값으로 사용
        file_path = os.path.join(dataset_dir, filename)

        file_info = {
            "cars": [],
            "accidents": []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(',')
                    label = parts[0]

                    if label == 'car':
                        # car, ID, x1, y1, x2, y2 형식
                        if len(parts) >= 6:
                            car_data = {
                                "id": int(parts[1]),
                                "bbox": [int(x) for x in parts[2:6]]
                            }
                            file_info["cars"].append(car_data)

                    elif label == 'A':
                        # A, VehicleID, StartFrame, EndFrame 형식
                        if len(parts) >= 4:
                            accident_data = {
                                "vehicle_id": int(parts[1]),
                                "start_frame": int(parts[2]),
                                "end_frame": int(parts[3])
                            }
                            file_info["accidents"].append(accident_data)

            combined_data[file_id] = file_info

        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({filename}): {e}")

    # JSON 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as json_f:
        json.dump(combined_data, json_f, indent=4, ensure_ascii=False)

    print(f"\n변환 완료!")
    print(f"결과 파일: {output_file}")


if __name__ == "__main__":
    convert_dataset_to_json()
