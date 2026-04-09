## vodreview

Tool to detect objects in a video. Optimized for longer content (VODs) and generic, everyday objects (e.g., chair, soda can) that can be read by YOLO-World. Also supports a Moondream-based pipeline for arbitrary prompts that YOLO-World can't handle. Flexible, but not super flexible.

## YOLO-World

`detect.py` runs the current YOLO-World pipeline:
- sample frames with ffmpeg
- detect the requested classes
- return timestamped detection records

### Clip formation

After `detect.py` gets the records with timestamps and confidence levels, we need a way to interpret the records and form clips (portions of the main video) from them.

## Moondream

`detect_moondream.py` runs the Moondream batch pipeline:
- sample frames with ffmpeg
- base64 encode each frame, write to a JSONL request file
- upload JSONL to Moondream's batch API
- poll until done, parse results into timestamped detection records

Cloud API. Needs `MOONDREAM_API_KEY` in the environment. No training, no scraping, no labeling. Takes any text prompt directly. Slower per frame than YOLO-World but no setup time, so better for shorter videos and niche objects YOLO-World doesn't know.

Output records use the same format as the YOLO-World pipeline, so `clips.py` works on both.

## Old pipeline
`old_version.py` is the legacy YOLOv8 + CLIP pipeline for mapping a free-form query to YOLO labels and scoring detections with CLIP.



