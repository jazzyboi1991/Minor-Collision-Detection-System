        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.inception1 = InceptionModule3D(192, 64, 96, 128, 16, 32, 32)
        self.inception2 = InceptionModule3D(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.inception3 = InceptionModule3D(480, 192, 96, 208, 16, 48, 64)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head_conv = nn.Conv3d(512, num_classes, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv3(self.conv2(x)))
        x = self.maxpool3(self.inception2(self.inception1(x)))
        x = self.avg_pool(self.inception3(x))
        logits = self.head_conv(x)
        return logits.view(logits.size(0), -1)


def crop_square_and_pad(frame, bbox, r_value, resize):
    h, w, _ = frame.shape
    x_min, y_min, x_max, y_max = bbox
    vw, vh = (x_max - x_min) * r_value, (y_max - y_min) * r_value
    cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2
    side = int(max(vw, vh))
    nx1, ny1 = cx - side // 2, cy - side // 2
    nx2, ny2 = cx + side // 2, cy + side // 2
    vx1, vy1 = max(0, nx1), max(0, ny1)
    vx2, vy2 = min(w, nx2), min(h, ny2)
    pl, pt = max(0, -nx1), max(0, -ny1)
    pr, pb = max(0, nx2 - w), max(0, ny2 - h)
    cropped = frame[vy1:vy2, vx1:vx2]
    if pl > 0 or pt > 0 or pr > 0 or pb > 0:
        cropped = np.pad(cropped, ((pt, pb), (pl, pr), (0, 0)), mode="constant")
    return cv2.resize(cropped, resize)


def read_bbox(txt_path, target_id):
    bboxes = {}
    with open(txt_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) >= 6 and parts[0] == "car":
                bboxes[int(parts[1])] = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
    if not bboxes:
        return None
    return bboxes.get(target_id, next(iter(bboxes.values())))


def label_from_filename(mp4_file):
    parts = os.path.splitext(mp4_file)[0].split("_")
    return 1 if len(parts) >= 2 and len(parts[1]) == 2 and parts[1][1] == "A" else 0


def predict_video(model, video_path, txt_path, device, target_id, r_value, resize):
    bbox = read_bbox(txt_path, target_id)
    if bbox is None:
        return None

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    tensor_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frames.append(transform(crop_square_and_pad(frame_rgb, bbox, r_value, resize)))
    cap.release()

    if not tensor_frames:
        return None

    while len(tensor_frames) < 30:
        tensor_frames.append(tensor_frames[-1])

    full_video_tensor = torch.stack(tensor_frames).permute(1, 0, 2, 3)
    predicted_label = 0
    for i in range(len(tensor_frames) - 29):
        clip = full_video_tensor[:, i:i + 30, :, :].unsqueeze(0).to(device)
        outputs = model(clip)
        pred_class = torch.argmax(F.softmax(outputs, dim=1), dim=1).item()
        if pred_class == 1:
            predicted_label = 1
            break
    return predicted_label


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HitAndRun3DCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    all_files = set(os.listdir(args.data_dir))
    pairs = sorted(f for f in all_files if f.endswith(".mp4") and f"{os.path.splitext(f)[0]}.txt" in all_files)
    if args.samples is not None:
        random.seed(args.seed)
        pairs = random.sample(pairs, min(args.samples, len(pairs)))

    total = correct = skipped = 0
    wrong = []
    with torch.no_grad():
        for mp4_file in pairs:
            base = os.path.splitext(mp4_file)[0]
            video_path = os.path.join(args.data_dir, mp4_file)
            txt_path = os.path.join(args.data_dir, f"{base}.txt")
            gt = label_from_filename(mp4_file)
            pred = predict_video(model, video_path, txt_path, device, args.target_id, args.r_value, (args.resize, args.resize))
            if pred is None:
                skipped += 1
                continue
            total += 1
            correct += int(pred == gt)
            if pred != gt:
                wrong.append((mp4_file, gt, pred))

    accuracy = (correct / total * 100) if total else 0.0
    print(f"total={total}")
    print(f"correct={correct}")
    print(f"skipped={skipped}")
    print(f"accuracy={accuracy:.2f}%")
    if wrong:
        print("wrong:")
        for name, gt, pred in wrong:
            gt_name = "Accident" if gt == 1 else "Normal"
            pred_name = "Accident" if pred == 1 else "Normal"
            print(f"- {name}: gt={gt_name}, pred={pred_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/leezungzoo/Desktop/AI-develop/Sample")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--r-value", type=float, default=1.0)
    parser.add_argument("--resize", type=int, default=224)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
