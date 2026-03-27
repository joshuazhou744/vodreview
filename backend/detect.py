# imports
import json
import subprocess
import numpy as np

from ultralytics import YOLO

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
def iter_frames_ffmpeg(path: str, fps: float = 2):
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

def detect_records_yolo(label, input_video_path, weights, fps=1, confidence=0.3):
    model = YOLO(weights)
    records = []

    for raw, w, h, time_s in iter_frames_ffmpeg(input_video_path, fps=fps):
        frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))
        frame = frame[:, :, ::-1]  # rgb -> bgr for ultralytics

        results = model(frame, conf=confidence, verbose=False)[0]

        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            records.append({
                "label": label,
                "confidence": float(box.conf),
                "bbox": tuple(xyxy),
                "timestamp_s": time_s,
                "timestamp": format_mmss(time_s),
            })

    print(f"[detect] found {len(records)} detections")
    return records
