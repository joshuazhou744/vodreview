## vodreview

Tool to detect objects in a video. Optimized for longer content (VODs) and generic, everyday objects (e.g., chair, soda can) that can be read by YOLO-World. Flexible, but not super flexible.

## YOLO-World

`detect.py` runs the current YOLO-World pipeline:
- sample frames with ffmpeg
- detect the requested classes
- return timestamped detection records

### Clip formation

After `detect.py` gets the records with timestamps and confidence levels, we need a way to interpret the records and form clips (portions of the main video) from them.

## Old pipeline
`old_version.py` is the legacy YOLOv8 + CLIP pipeline for mapping a free-form query to YOLO labels and scoring detections with CLIP.