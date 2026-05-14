import cv2
import os
import glob
from tqdm import tqdm


def flip_videos(directory):
    """
    지정된 디렉토리 내의 모든 .mp4 파일을 찾아 좌우 반전(Horizontal Flip) 시킵니다.
    """
    # 이전 실행에서 남은 임시 파일 제거
    for temp_file in glob.glob(os.path.join(directory, "*.temp.mp4")):
        try:
            os.remove(temp_file)
        except:
            pass

    # .mp4 확장자를 가진 모든 파일 목록 가져오기 (임시 파일 제외)
    mp4_files = [f for f in glob.glob(os.path.join(
        directory, "*.mp4")) if not f.endswith(".temp.mp4")]

    if not mp4_files:
        print(f"'{directory}' 경로에서 처리할 mp4 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(mp4_files)}개의 영상을 처리합니다.")

    for video_path in tqdm(mp4_files, desc="영상을 반전하는 중"):
        # 임시 출력 파일 경로 (처리 중 원본 보호를 위해)
        temp_output = video_path + ".temp.mp4"

        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"영상을 열 수 없습니다: {video_path}")
            continue

        # 비디오 속성 정보 가져오기
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 코덱 설정 (H.264 - avc1 사용으로 브라우저 호환성 확보)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        # 비디오 작성 객체 생성
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 좌우 반전 (1: 좌우, 0: 상하, -1: 좌우상하 모두)
            flipped_frame = cv2.flip(frame, 1)

            # 반전된 프레임 저장
            out.write(flipped_frame)

        # 객체 해제
        cap.release()
        out.release()

        # 원본 파일을 반전된 임시 파일로 교체 (파일 덮어쓰기)
        if os.path.exists(temp_output):
            os.replace(temp_output, video_path)


if __name__ == "__main__":
    # 스크립트 파일의 위치를 기준으로 상위 폴더(Reversed/) 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(os.path.dirname(current_dir), "mp4")

    flip_videos(target_dir)
