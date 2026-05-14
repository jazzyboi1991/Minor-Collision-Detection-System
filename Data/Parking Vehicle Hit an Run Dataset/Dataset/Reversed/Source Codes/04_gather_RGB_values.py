import cv2
import numpy as np
import os
import json
from tqdm import tqdm


def calculate_video_rgb_average(video_path):
    """
    영상 파일의 모든 프레임을 읽어 전체 RGB 평균값을 계산합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 영상을 열 수 없습니다. {video_path}")
        return None

    frame_means = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # cv2.mean()은 (B, G, R, A) 순서로 평균값을 반환합니다.
        # [0:3]을 슬라이싱하여 B, G, R만 가져오고 이를 R, G, B 순서로 바꿉니다.
        mean_bgr = cv2.mean(frame)[:3]
        mean_rgb = [mean_bgr[2], mean_bgr[1], mean_bgr[0]]
        frame_means.append(mean_rgb)

    cap.release()

    if not frame_means:
        return None

    # 모든 프레임의 평균값을 다시 평균하여 영상 전체의 RGB 평균 산출
    global_avg_rgb = np.mean(frame_means, axis=0).tolist()
    return global_avg_rgb


def main():
    # 현재 스크립트 위치 기준 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 데이터셋 위치를 부모 디렉토리(Reversed)로 설정
    dataset_dir = os.path.join(os.path.dirname(base_dir), "mp4")
    output_file = os.path.join(base_dir, "reversed_average_RGB_values.json")

    if not os.path.exists(dataset_dir):
        print(f"Error: 데이터셋 디렉토리를 찾을 수 없습니다: {dataset_dir}")
        return

    # mp4 파일 목록 추출 (임시 파일 제외)
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(
        '.mp4') and not f.endswith('.temp.mp4')]
    video_files.sort()

    results = {}

    print(f"총 {len(video_files)}개의 영상을 처리합니다.")

    for video_name in tqdm(video_files, desc="RGB 값 추출 중"):
        video_path = os.path.join(dataset_dir, video_name)
        avg_rgb = calculate_video_rgb_average(video_path)

        if avg_rgb:
            # 확장자를 제외한 파일명을 키로 사용하도록 수정 (선택 사항이나 병합 시 용이)
            file_id = os.path.splitext(video_name)[0]
            results[file_id] = {
                "r_avg": round(avg_rgb[0], 2),
                "g_avg": round(avg_rgb[1], 2),
                "b_avg": round(avg_rgb[2], 2)
            }

    # 결과를 JSON 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n작업 완료! 결과가 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()
