import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
import os
import glob
import yaml
import cv2

MODEL_ID = "yolo_world/l"
model = YOLOWorld(model_id=MODEL_ID)

def auto_detect(image_dir, label, confidence_threshold=0.5, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(image_dir)

    model.set_classes([label])

    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(image_dir, "*")))
    labeled = 0
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        results = model.infer(img, confidence=confidence_threshold)
        det = sv.Detections.from_inference(results)
        if len(det) == 0:
            continue

        # write YOLO format label file (bbox and confidence)
        txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_name)
        with open(txt_path, "w") as f:
            for xyxy in det.xyxy:
                x1, y1, x2, y2 = xyxy
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / w
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        labeled += 1
    
    print(f"[label] labeled {labeled}/{len(images)} images")

    # make dataset.yaml spec
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    dataset_config = {
        "path": os.path.abspath(output_dir),
        "train": "images",
        "val": "images",
        "names": {0: label},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f)

    print(f"[label] dataset.yaml written to {yaml_path}")
    return yaml_path