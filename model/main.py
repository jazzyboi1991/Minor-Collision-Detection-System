import argparse
import torch

import config
from device_utils import get_device, is_cuda_like
from model import HitAndRun3DCNN


def _load_model(weights_path):
    device = get_device()
    model = HitAndRun3DCNN(num_classes=config.MODEL_NUM_CLASSES).to(device)
    if is_cuda_like(device) and config.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last_3d)
    try:
        # DirectML은 map_location 직접 지원이 불안정하므로 CPU 경유 로드
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"가중치 로드 완료: {weights_path}")
    except FileNotFoundError:
        print(f"에러: 가중치 파일을 찾을 수 없습니다 -> {weights_path}")
        raise
    return model, device


def run_train():
    from train import train_model
    if not config.DATA_DIR.exists():
        print(f"에러: 데이터 폴더를 찾을 수 없습니다 -> {config.DATA_DIR}")
        return
    print(f"데이터 디렉토리 확인 완료: {config.DATA_DIR}. 학습을 시작합니다!")
    train_model(
        data_dir=config.DATA_DIR,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_epochs=config.TRAIN_NUM_EPOCHS,
        clip_length=config.CLIP_LENGTH,
        r_value=config.R_VALUE,
        resize=config.RESIZE,
        save_path=config.TRAIN_BEST_MODEL_SAVE_PATH,
        train_split_ratio=config.TRAIN_SPLIT_RATIO,
        early_stopping_patience=config.TRAIN_EARLY_STOPPING_PATIENCE,
        use_amp=config.USE_AMP,
        use_channels_last=config.USE_CHANNELS_LAST,
    )


def run_predict():
    from predict_cam import predict_hit_and_run_final
    model, _ = _load_model(config.PREDICT_WEIGHTS_PATH)
    predict_hit_and_run_final(
        model,
        config.PREDICT_VIDEO_PATH,
        config.PREDICT_TXT_PATH,
        target_id=config.TARGET_ID,
        r_value=config.R_VALUE,
        resize=config.RESIZE,
        output_dir=config.PREDICT_OUTPUT_DIR,
        infer_batch_size=config.PREDICT_INFER_BATCH_SIZE,
        window_stride=config.PREDICT_WINDOW_STRIDE,
    )


def run_eval():
    from evaluate import evaluate_folder_accuracy
    model, _ = _load_model(config.EVAL_WEIGHTS_PATH)
    evaluate_folder_accuracy(
        model,
        config.EVAL_FOLDER_PATH,
        target_id=config.TARGET_ID,
        r_value=config.R_VALUE,
        resize=config.RESIZE,
        num_samples=config.EVAL_NUM_SAMPLES,
        infer_batch_size=config.EVAL_INFER_BATCH_SIZE,
        window_stride=config.EVAL_WINDOW_STRIDE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hit-and-Run 3D-CNN 실행 스크립트")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "predict", "eval"],
        help="실행 모드: train(학습) / predict(단일 영상 CAM예측) / eval(정확도 평가)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "predict":
        run_predict()
    elif args.mode == "eval":
        run_eval()
