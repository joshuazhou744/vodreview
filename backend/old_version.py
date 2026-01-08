import json
import subprocess
import sys
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import clip
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("yolov8n.pt")

CLIP_BATCH_SIZE = 64
buffer = []

def flush_buffer(buffer, query_embedding):
    if not buffer:
        return []
    
    clip_inputs = [clip_preprocess(Image.fromarray(d["crop"])) for d in buffer]
    clip_batch = torch.stack(clip_inputs).to(device)

    with torch.inference_mode():
        image_embedding = clip_model.encode_image(clip_batch)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        sims = (image_embedding @ query_embedding.T).squeeze(1).cpu().numpy()

    results = []
    for d, s in zip(buffer, sims):
        results.append({
            "label": d["label"],
            "yolo_conf": d["yolo_conf"],
            "crop": d["crop"],
            "clip_similarity": float(s),
            "timestamp": d["timestamp"],
        })
    return results

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

def process_frame(frame, similar_labels, timestamp):
    results = yolo_model(frame, conf=0.3, verbose=False)[0]
    
    detections = []

    for box in results.boxes:
        # get class label
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]

        # don't process if label not similar
        if label not in similar_labels:
            continue

        # get border box of the detected object
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # get confidence score
        conf = float(box.conf[0])

        # crop image
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # append this detected object
        detections.append({
            "crop": crop,
            "label": label,
            "yolo_conf": conf,
            "timestamp": timestamp
        })

    return detections

def score_with_clip(detections, query_embedding, timestamp):
    clip_inputs = [clip_preprocess(Image.fromarray(d["crop"])) for d in detections]
    clip_batch = torch.stack(clip_inputs).to(device)
    with torch.inference_mode():
        image_emb = clip_model.encode_image(clip_batch)
        image_emb /= image_emb.norm(dim=-1, keepdim=True)
        sims = (image_emb @ query_embedding.T).squeeze(1).cpu().numpy()

    results = []
    for d, s in zip(detections, sims):
        results.append({
            "label": d["label"],
            "yolo_conf": d["yolo_conf"],
            "crop": d["crop"],
            "clip_similarity": float(s),
            "timestamp": timestamp,
        })
    return results

def compute_similar_labels(labels, label_embeddings, query_embedding, k=3):
    sims = (query_embedding @ label_embeddings.T).squeeze(0)
    topk = torch.topk(sims, k=k).indices.tolist()
    return { labels[i] for i in topk }

def get_records_df(records):
    return pd.DataFrame.from_records(records).sort_values(by=["clip_similarity"], ascending=False).reset_index(drop=True)

def main():
    user_query = "glasses"
    video_path = "video3.mp4"
    k = 3
    
    records = []
    frame_count = 0

    with torch.inference_mode():
        text_tokens = clip.tokenize([user_query]).to(device)
        query_embedding = clip_model.encode_text(text_tokens)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

        labels = list(yolo_model.names.values())    
        label_tokens = clip.tokenize(labels).to(device)
        label_embeddings = clip_model.encode_text(label_tokens)
        label_embeddings /= label_embeddings.norm(dim=-1, keepdim=True)

        similar_labels = compute_similar_labels(labels, label_embeddings, query_embedding, k)

    for _raw, _w, _h, _ts, _src_fps in iter_frames_ffmpeg(video_path, fps=2):
        frame = np.frombuffer(_raw, np.uint8).reshape((_h, _w, 3))

        detections = process_frame(frame, similar_labels, _ts)

        # if detections:
        #     scored = score_with_clip(detections, query_embedding, _ts)
        #     records.extend(scored)

        if detections:
            buffer.extend(detections)
            if len(buffer) >= CLIP_BATCH_SIZE:
                records.extend(flush_buffer(buffer, query_embedding))

        frame_count += 1
    
    # flush leftover buffer
    records.extend(flush_buffer(buffer, query_embedding))
    
    
    records_df = get_records_df(records)
    print(records_df)
    print(f"Read {frame_count} frames.")


if __name__ == "__main__":
    import time
    start = time.perf_counter()

    main()

    end = time.perf_counter()
    print(f"Elapsed: {end - start:.2f}s")

