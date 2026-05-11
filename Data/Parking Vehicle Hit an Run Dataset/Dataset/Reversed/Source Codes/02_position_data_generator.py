import os
import glob


def generate_reversed_annotations(src_dir, dst_dir, width=1920):
    """
    Normal 데이터셋의 .txt 파일을 읽어 좌우 반전된 좌표를 계산한 뒤,
    Reversed 데이터셋 폴더에 새로운 .txt 파일을 생성합니다.
    """
    # 원본(.txt) 파일 목록 가져오기
    txt_files = glob.glob(os.path.join(src_dir, "*.txt"))

    if not txt_files:
        print(f"'{src_dir}' 경로에서 txt 파일을 찾을 수 없습니다.")
        return

    # 출력 폴더가 없으면 생성
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    print(f"총 {len(txt_files)}개의 어노테이션 파일을 처리합니다.")

    for src_path in txt_files:
        # 파일명 추출 및 '_Reversed' 접미사 추가
        filename = os.path.basename(src_path)
        base_name, ext = os.path.splitext(filename)
        dst_filename = f"{base_name}_Reversed{ext}"
        dst_path = os.path.join(dst_dir, dst_filename)

        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"파일을 읽는 중 오류 발생 ({src_path}): {e}")
            continue

        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            # 차량 위치 정보(car)인 경우 좌표 변환 수행
            if parts[0] == 'car' and len(parts) >= 6:
                try:
                    car_id = parts[1]
                    x1 = int(parts[2])
                    y1 = int(parts[3])
                    x2 = int(parts[4])
                    y2 = int(parts[5])

                    # 좌우 반전 좌표 계산 (1920px 기준)
                    # 원본 x1(좌측) -> 1920 - x1 (우측)
                    # 원본 x2(우측) -> 1920 - x2 (좌측)
                    # 따라서 새로운 x1은 width - x2, 새로운 x2는 width - x1이 됩니다.
                    new_x1 = width - x2
                    new_x2 = width - x1

                    new_lines.append(
                        f"car,{car_id},{new_x1},{y1},{new_x2},{y2}")
                except ValueError:
                    print(f"좌표 변환 중 오류 발생 (라인: {line})")
                    new_lines.append(line)
            else:
                # 사고 정보(A)나 기타 라인은 그대로 유지
                new_lines.append(line)

        # 변환된 내용을 Reversed 폴더에 저장
        try:
            with open(dst_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines) + '\n')
        except Exception as e:
            print(f"파일 저장 중 오류 발생 ({dst_path}): {e}")


if __name__ == "__main__":
    # 스크립트 파일 위치를 기준으로 경로 자동 설정
    # .../Dataset/Reversed/Utility/02_position_data_generator.py
    current_utility_dir = os.path.dirname(
        os.path.abspath(__file__))  # Utility 폴더
    reversed_dir = os.path.dirname(
        current_utility_dir)             # Reversed 폴더
    dataset_dir = os.path.dirname(reversed_dir)                    # Dataset 폴더
    normal_txt_dir = os.path.join(
        dataset_dir, "Normal", "txt")               # Normal/txt 폴더
    reversed_txt_dir = os.path.join(
        reversed_dir, "txt")                     # Reversed/txt 폴더

    # 원본(Normal)에서 읽어서 대상(Reversed) 폴더에 생성
    generate_reversed_annotations(normal_txt_dir, reversed_txt_dir)
    print("모든 어노테이션 파일의 좌우 반전 변환이 완료되었습니다.")
