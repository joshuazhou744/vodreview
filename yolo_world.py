# simplification of main.py using yolo world rather than yolo + clip
# ** requires a gpu for reasonable performance **
# uses requirements.txt packages

# imports
from tqdm import tqdm
import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

import json
import subprocess
import numpy as np
import pandas as pd
import time

def ffprobe_video(path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return data["streams"][0]


def parse_fps(avg_frame_rate: str) -> float:
    num, den = avg_frame_rate.split("/")
    return float(num) / float(den)


def iter_frames_ffmpeg(path: str, fps: float = 1):
    info = ffprobe_video(path)
    width = int(info["width"])
    height = int(info["height"])
    _fps = parse_fps(info["avg_frame_rate"])

    frame_size = width * height * 3  # rgb24
    cmd = [
        "ffmpeg",
        "-i",
        path,
        "-vf",
        f"fps={fps}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    idx = 0
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        timestamp = idx / float(fps)
        yield raw, width, height, timestamp, _fps
        idx += 1

    proc.stdout.close()
    proc.wait()


SOURCE_VIDEO = f"./video2.mp4"
OUTPUT_VIDEO = f"./output.mp4"

CLASSES = ["person"] # define classes
CONFIDENCE = 0.08 # confidence threshold
NMS = 0.3 # non-max suppression threshold
MODEL_ID = "yolo_world/l" # model id

model = YOLOWorld(model_id=MODEL_ID)
model.set_classes(CLASSES)

start = time.perf_counter()

records = []

for raw, w, h, ts, _src_fps in iter_frames_ffmpeg(SOURCE_VIDEO, fps=2):
    frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))  # rgb24

    # If YOLO‑World expects BGR (it likely does), convert:
    frame = frame[:, :, ::-1]

    results = model.infer(frame, confidence=CONFIDENCE)
    det = sv.Detections.from_inference(results).with_nms(NMS)

    for xyxy, conf, cls in zip(det.xyxy, det.confidence, det.class_id):
        records.append({
            "label": CLASSES[int(cls)],
            "confidence": float(conf),
            "bbox": tuple(map(float, xyxy)),
            "timestamp": ts,
        })

end = time.perf_counter()
print(f"Elapsed: {end - start:.2f}s")

if records:
  records_df = pd.DataFrame(records)
  td = pd.to_timedelta(records_df["timestamp"], unit="s")
  records_df["timestamp_mmss"] = (
      td.dt.components["minutes"].astype(str).str.zfill(2)
      + ":" +
      td.dt.components["seconds"].astype(str).str.zfill(2)
  )
  print(records_df.sort_values(by=["confidence"], ascending=False))