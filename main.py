# simplification of main.py using yolo world rather than yolo + clip
# ** requires a gpu for reasonable performance **
# uses requirements.txt packages

# imports
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

import json
import subprocess
import numpy as np
import pandas as pd

MODEL_ID = "yolo_world/m" # model id

model = YOLOWorld(model_id=MODEL_ID)

def download_video():
    # TODO: implenent video download based on the object storage service i use
    pass

# get video info
def ffprobe_video(path: str) -> dict:
    # return info about video
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height", # width and height info
        "-of",
        "json", # output format
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe returned no video streams for: {path}")
    return streams[0]


# iterate frames from video using ffmpeg
def iter_frames_ffmpeg(path: str, fps: float = 1):
    if fps <= 0:
        raise ValueError("fps must be > 0")
    info = ffprobe_video(path)
    width = int(info["width"])
    height = int(info["height"])

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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe was not created")

    idx = 0
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        time_s = idx / float(fps)
        yield raw, width, height, time_s
        idx += 1

    proc.stdout.close()
    
    # check ffmpeg error
    stderr = proc.stderr.read()
    proc.stderr.close()
    code = proc.wait()
    if code != 0:
        err_text = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed with code {code}: {err_text}")

# format seconds to mm:ss
def format_mmss(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"

def main(
        classes: list[str],
        input_video_path: str,
        confidence: float = 0.08,
        nms: float = 0.3,
        debug_level: int = 0,
    ):
    if not classes:
        raise ValueError("classes must contain at least one label")
    model.set_classes(classes)

    records = []

    for raw, w, h, time_s in iter_frames_ffmpeg(input_video_path, fps=2):
        frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))

        results = model.infer(frame, confidence=confidence)
        det = sv.Detections.from_inference(results).with_nms(nms)

        for xyxy, conf, cls in zip(det.xyxy, det.confidence, det.class_id):
            records.append({
                "label": classes[int(cls)],
                "confidence": float(conf),
                "bbox": tuple(map(float, xyxy)),
                "timestamp_s": time_s,
                "timestamp": format_mmss(time_s),
            })

    # print out records sorted by confidence
    if records and debug_level == 1:
        records_df = pd.DataFrame.from_records(records)
        print(records_df.sort_values(by=["confidence"], ascending=False))
    elif debug_level == 1:
        print("No detections found.")

    return records

if __name__ == "__main__":
    classes = ["person"]
    input_video_path = "video.mp4"
    main(classes, input_video_path, debug_level=1)
