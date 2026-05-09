import cv2
import numpy as np
import os
import json
from tqdm import tqdm


def gather_brightness_values():
    # 스크립트 파일의 위치를 기준으로 비디오 파일들이 있는 디렉토리 경로 설정
    # Utility 폴더 안에 있으므로 부모 디렉토리가 Normal 디렉토리임
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(os.path.dirname(script_dir), "mp4")
    output_path = os.path.join(script_dir, "average_brightness_values.json")

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files.sort()

    brightness_data = {}

    print(f"총 {len(video_files)}개의 영상을 처리합니다.")

    for video_name in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"경고: {video_name} 파일을 열 수 없습니다.")
            continue

        total_brightness = 0.0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임을 그레이스케일로 변환하여 명도 계산
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            total_brightness += np.mean(gray)
            frame_count += 1

        cap.release()

        # 전체 프레임의 평균 명도 계산
        if frame_count > 0:
            brightness_data[video_name] = float(total_brightness / frame_count)
        else:
            brightness_data[video_name] = 0.0

    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(brightness_data, f, indent=4, ensure_ascii=False)

    print(f"\n작업 완료! 결과가 {output_path}에 저장되었습니다.")


if __name__ == "__main__":
    gather_brightness_values()
