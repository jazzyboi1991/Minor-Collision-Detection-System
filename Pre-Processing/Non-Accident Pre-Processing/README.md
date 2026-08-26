# 비사고 영상 자동 어노테이션

논문의 비사고 클래스 `S`처럼, 주행 차량 주변의 인접 차량 2대를 `car 0`, `car 1`로 기록하는 파이프라인이다. 자동 탐지 결과는 초안이며, 기존 수동 GUI에서 검수한 뒤 최종 파일을 만든다.

## 설치

저장소 루트에서 실행한다.

```bash
cd "/Users/manuelpark/Documents/대학교 프로그래밍/캡스톤디자인"
source "Minor-Collision-Detection-System/.venv/bin/activate"
python -m pip install -r "Real Data/Pre-Processing/Non-Accident Pre-Processing/requirements.txt"
brew install ffmpeg
```

## 자동 초안 생성

```bash
cd "Real Data/Pre-Processing/Non-Accident Pre-Processing"
source "../../../Minor-Collision-Detection-System/.venv/bin/activate"

python -m src.nonaccident_pipeline inventory \
  --source-root output/normal/MP4 \
  --output work/inventory.json

python -m src.nonaccident_pipeline detect \
  --inventory work/inventory.json \
  --output work/drafts.json \
  --model yolo11n.pt

python -m src.nonaccident_pipeline prepare-review \
  --drafts work/drafts.json \
  --output work/nonaccident_annotations.json
```

`detect`는 Ultralytics YOLO와 ByteTrack으로 여러 프레임의 승용차를 추적한다. 기본 탐지 클래스는 COCO의 `car`(`class 2`)만 사용하며, 신뢰도 `0.5` 미만의 탐지는 제외한다. 이동량이 큰 주행 차량은 제외하고, 그 주변에 있으면서 정지 상태인 차량 2대를 기준 차량으로 제안한다.

가림 대응도 자동으로 적용된다. 기준 박스는 두 차량이 동시에 검출된 프레임 중에서 선택하며, 이동 차량의 박스와 크게 겹치는 프레임은 대표 프레임 후보에서 감점한다. 짧게만 나타난 추적 ID도 기준 차량 후보에서 제외한다. 따라서 이동 차량이 앞을 지나가는 순간의 헤드라이트·차체 일부 박스가 기준 박스로 선택될 가능성을 줄인다.

코드나 탐지 조건을 변경한 뒤에는 기존 `drafts.json`을 재사용하지 말고 `detect`부터 다시 실행한다.

```bash
python -m src.nonaccident_pipeline detect \
  --inventory work/inventory.json \
  --output work/drafts.json \
  --model yolo11n.pt
```

## JSON 포맷 정리

`drafts.json`을 프로젝트의 `../../../Minor-Collision-Detection-System/.prettierrc.json` 설정에 맞춰 정리하려면 다음 명령어를 실행한다.

```bash
npx prettier \
  --config "../../../Minor-Collision-Detection-System/.prettierrc.json" \
  --write work/drafts.json
```

검수용 어노테이션 파일도 같은 방식으로 포맷한다.

```bash
npx prettier \
  --config "../../../Minor-Collision-Detection-System/.prettierrc.json" \
  --write work/nonaccident_annotations.json
```

두 JSON 파일을 한 번에 포맷하려면 다음 명령어를 사용한다.

```bash
npx prettier \
  --config "../../../Minor-Collision-Detection-System/.prettierrc.json" \
  --write work/drafts.json work/nonaccident_annotations.json
```

Prettier가 설치되어 있지 않은 경우에는 `npx --yes prettier`를 사용한다.

```bash
npx --yes prettier \
  --config "../../../Minor-Collision-Detection-System/.prettierrc.json" \
  --write work/drafts.json
```

## JSON 무결성 검사 및 메타데이터 복구

`nonaccident_annotations.json`의 구조·박스·기준 프레임과 `source_video`가 가리키는 실제 MP4의 해상도·FPS·프레임 수를 검사한다.

```bash
python -m src.nonaccident_pipeline validate \
  --annotations work/nonaccident_annotations.json
```

실제 MP4와 JSON의 미디어 정보가 다르면, 수정 전 JSON을 백업한 뒤 실제 영상 기준으로 미디어 필드만 갱신한다. 박스 좌표·차량 ID·`events`·`status`는 변경하지 않는다. 단, `reference_frame`이 현재 영상 범위를 벗어난 경우에는 영상의 마지막 프레임으로 보정한다. 백업 파일은 덮어쓰지 않으므로 이미 같은 이름이 있으면 다른 파일명을 지정한다.

```bash
python -m src.nonaccident_pipeline repair-metadata \
  --annotations work/nonaccident_annotations.json \
  --backup work/nonaccident_annotations.backup1.json
```

복구 후 다시 검사한다.

```bash
python -m src.nonaccident_pipeline validate \
  --annotations work/nonaccident_annotations.json
```

영상 파일이 없거나 박스 좌표·차량 ID·기준 프레임이 잘못된 경우에는 자동으로 박스를 고치지 않고 오류로 보고한다. 해당 영상은 GUI에서 검수한 뒤 저장한다.

## GUI 검수

프로젝트 루트(`캡스톤디자인`)에서 실행하는 경우, 먼저 `src` 패키지가 있는 작업 폴더로 이동해야 합니다.

```bash
cd "Real Data/Pre-Processing/Non-Accident Pre-Processing"
source "../../../Minor-Collision-Detection-System/.venv/bin/activate"

python -m src.app \
  --mode non-accident \
  --source-root output/normal/MP4 \
  --annotations work/nonaccident_annotations.json \
  --output-root output
```

`python -m src.app` 명령은 반드시 위 폴더에서 실행해야 합니다. 프로젝트 루트에서 바로 실행하면 현재 폴더에 `src` 패키지가 없기 때문에 다음 오류가 발생합니다.

```text
ModuleNotFoundError: No module named 'src'
```

이미 `Real Data/Pre-Processing/Non-Accident Pre-Processing` 폴더에 있는 경우에는 `cd` 명령을 다시 실행하지 않고 가상환경 활성화와 Python 명령만 실행하면 됩니다.

비사고 모드에서는 사고 이벤트와 시작·종료 프레임을 입력하지 않는다. 자동 박스를 수정하고, 기준 차량이 정확히 두 대인지 확인한 뒤 `기준 차량 2대 확정`과 `저장`을 누른다. 이동 차량은 최종 TXT에 포함하지 않는다.

영상 확대·이동:

- `+`: 확대
- `−`: 축소
- `원래대로`: 확대 배율과 이동 위치를 100% 기본 화면으로 복원
- 마우스 휠: 커서 위치를 중심으로 확대·축소
- 확대 상태에서 마우스 휠 버튼을 누른 채 드래그: 화면 이동

## 박스 상태 분류

`nonaccident_annotations.json`은 별도 태그 없이 `status` 하나로 상태를 기록한다.

- `status: "normal"`: `car 0`, `car 1` 두 박스가 모두 있고 좌표·프레임·영상 경계 검증을 통과한 상태
- `status: "needs_review"`: 박스가 비어 있거나, 하나만 있거나, ID·좌표·프레임 검증에 실패한 상태

`normal`은 박스 형식 검증을 통과했다는 뜻이다. 기존 `confirmed` 값도 호환되지만, 비사고 영상에서는 `normal`을 사용한다.

GUI에서 `영상 현황`을 직접 선택한 뒤 `저장`을 누르면 선택한 `status` 값이 그대로 JSON에 저장된다. 저장 시 박스 개수에 따라 상태를 자동으로 덮어쓰지 않는다.

이미 생성된 JSON에 태그를 다시 계산하려면 다음 명령어를 실행한다.

```bash
python -m src.nonaccident_pipeline classify \
  --annotations work/nonaccident_annotations.json \
  --output work/nonaccident_annotations.classified.json
```

원본 파일을 갱신하려면 출력 경로를 입력 파일과 같게 지정할 수 있다.

```bash
python -m src.nonaccident_pipeline classify \
  --annotations work/nonaccident_annotations.json \
  --output work/nonaccident_annotations.json
```

분류 결과는 터미널에 `normal`과 `needs_review` 개수로 표시된다. `needs_review` 영상은 GUI에서 박스를 추가·수정한 뒤 저장하고, 다시 `classify`를 실행한다.

## 최종 내보내기

```bash
python -m src.nonaccident_pipeline export \
  --annotations work/nonaccident_annotations.json \
  --output-root output
```

`normal` 또는 기존 호환 상태인 `confirmed` 영상만 처리한다. `needs_review`와 `excluded`는 건너뛴다.

TXT는 논문 형식에 맞춰 `A` 레코드 없이 두 줄만 생성한다.

```text
car,0,x1,y1,x2,y2
car,1,x1,y1,x2,y2
```

결과는 `output/normal/txt/`와 `output/normal/visualized/`에 저장된다. 검수 MP4는 원본의 전체 길이·해상도·FPS·프레임 수를 유지한다.

## 주의

- 원본 `Real Data/Non-Accident/Edit.aep_AME/*.mp4`는 수정하지 않는다.
- `yolo11n.pt`는 첫 탐지 실행 때 자동으로 다운로드될 수 있다.
- 자동 결과는 최종 확정본이 아니므로 GUI 검수를 거쳐야 한다.
