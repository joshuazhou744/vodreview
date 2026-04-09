import os
import json
import base64
import time
import subprocess
import requests
import numpy as np
from PIL import Image
from io import BytesIO

from detect import iter_frames_ffmpeg, format_mmss


def _video_duration_s(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def _format_hms(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# 1. Extract frames using iter_frames_ffmpeg
# 2. Base64 encode each frame
# 3. Build a JSONL file (one detect request per line)
# 4. Upload JSONL to Moondream batch API
# 5. Poll until complete
# 6. Download and parse results into records

def detect_records_moondream(label, input_video_path, fps=1, api_key=None):
    if api_key is None:
        api_key = os.environ.get("MOONDREAM_API_KEY")
    if not api_key:
        raise ValueError("no moondream api key")

    start_wall = time.perf_counter()
    video_duration = _video_duration_s(input_video_path)

    BASE_URL = "https://api.moondream.ai/v1/batch"
    headers = {"X-Moondream-Auth": api_key}
    CHUNK_SIZE = 50 * 1024 * 1024 # 50mb chunks
    
    # extract frames and build jsonl file of all frames in base64
    jsonl_path = "batch_request.jsonl"
    frame_count = 0

    with open(jsonl_path, "w") as f:
        for raw, w, h, time_s in iter_frames_ffmpeg(input_video_path, fps=fps):
            frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))
            img = Image.fromarray(frame)

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            # moondream batch api request format in jsonl
            line = {
                "id": f"frame_{time_s}",
                "skill": "detect",
                "image": b64,
                "object": label,
            }
            f.write(json.dumps(line) + "\n")
            frame_count += 1
    
    print(f"[moondream] extracted {frame_count} frames to {jsonl_path}")

    # init multipart upload
    init = requests.post(f"{BASE_URL}?action=mpu-create", headers=headers).json()
    file_id = init["fileId"]
    upload_id = init["uploadId"]

    # upload in chunks
    parts = []
    with open(jsonl_path, "rb") as f:
        part_number = 1
        while chunk := f.read(CHUNK_SIZE): # walrus operator assigns and returns
            part = requests.put(
                f"{BASE_URL}/{file_id}?action=mpu-uploadpart&uploadId={upload_id}&partNumber={part_number}",
                headers={**headers, "Content-Type": "application/octet-stream"},
                data=chunk,
            ).json()
            parts.append({"partNumber": part_number, "etag": part["etag"]})
            part_number += 1

    # complete upload and process
    batch = requests.post(
        f"{BASE_URL}/{file_id}?action=mpu-complete&uploadId={upload_id}",
        headers=headers,
        json={"parts": parts},
    ).json()
    batch_id = batch["id"]
    print(f"[moondream] batch submitted: {batch['id']}")

    # poll until complete
    while True:
        status = requests.get(f"{BASE_URL}/{batch_id}", headers=headers).json()
        state = status["status"]
        progress = status.get("progress", {})
        print(f"Status: {status['status']}, Progress: {status['progress']}")
        if state in ("completed", "failed"):
            break
        time.sleep(10)

    if state == "failed":
        print(f"[moondream] batch failed: {status}")
        return []
    
    # download and parse results
    records = []
    for output in status.get("outputs", []):
        res = requests.get(output["url"])
        for line in res.text.strip().split("\n"):
            result = json.loads(line)
            if result["status"] != "ok":
                continue
            time_s = float(result["id"].replace("frame_", ""))
            for obj in result.get("result", {}).get("objects", []):
                bbox = (obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"])
                records.append({
                    "label": label,
                    "confidence": 1.0,
                    "bbox": bbox,
                    "timestamp_s": time_s,
                    "timestamp": format_mmss(time_s),
                })
    print(f"[moondream] found {len(records)} detections")

    elapsed = time.perf_counter() - start_wall
    ratio = elapsed / video_duration if video_duration > 0 else 0
    print(
        f"[moondream] elapsed {_format_hms(elapsed)} | "
        f"video duration {_format_hms(video_duration)} | "
        f"{ratio:.2f}x realtime"
    )

    return records

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("video_path")
    parser.add_argument("--fps", type=float, default=1)
    args = parser.parse_args()

    records = detect_records_moondream(args.label, args.video_path, fps=args.fps)
    df = pd.DataFrame(records)
    df.to_csv(f"{args.label}_moondream_records.csv", index=False)
    print(f"Saved {len(records)} records")
    
