from ultralytics import YOLO
import sys
import glob

weights = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/runs/train/weights/best.pt"
image_dir = sys.argv[2] if len(sys.argv) > 2 else "datasets/red_bull_can/images"

model = YOLO(weights)
images = sorted(glob.glob(f"{image_dir}/*"))[:10]

for img in images:
    results = model(img, conf=0.1, verbose=False, save=True, project="test_results")[0]
    n = len(results.boxes)
    if n > 0:
        dets = ", ".join(f"conf={float(b.conf):.3f} bbox={b.xyxy[0].tolist()}" for b in results.boxes)
        print(f"{img}: {n} detections — {dets}")
    else:
        print(f"{img}: no detections")
