def parse_records_for_clips(
        records: list[dict],
        fps: float,
        max_gap_s: float = 2.0,
        min_clip_duration: float = 0.5,
        max_clip_duration: float = 20.0,  # currently unused, reserved for future split/trim logic
        min_records_in_clip: int = 2,
    ) -> dict:
    # dictionary to store clips
    clips = {}

    # get records of each label
    label_groups = {}
    for r in records:
        label_groups.setdefault(r["label"], []).append(r)

    # iterate over each label and find clips by temporal data
    for label, items in label_groups.items():
        # sort records of each label by time
        items.sort(key=lambda r: r["timestamp_s"])
        # store the clips that we find
        label_clips = []

        # start the current cluster of records (building candidate clip) with the first record
        current_cluster = [items[0]]
        # only track the previous timestamp since that's all we compare against
        prev_ts = items[0]["timestamp_s"]

        # iterate over the rest of the records for the label
        for item in items[1:]:
            item_ts = item["timestamp_s"]
            if item_ts - prev_ts <= max_gap_s:
                current_cluster.append(item)
            else:
                clip = build_clip(current_cluster, fps, min_clip_duration, min_records_in_clip)
                if clip is not None:
                    label_clips.append(clip)
                current_cluster = [item]
            # save current item ts as prev_ts so next iter compares against it
            prev_ts = item_ts

        # check for leftover cluster
        clip = build_clip(current_cluster, fps, min_clip_duration, min_records_in_clip)
        if clip is not None:
            label_clips.append(clip)

        clips[label] = label_clips
    return clips

def build_clip(cluster, fps, min_clip_duration, min_records_in_clip):
    if len(cluster) < min_records_in_clip:
        return None

    start_s = cluster[0]["timestamp_s"]
    end_s = cluster[-1]["timestamp_s"] + (1.0 / fps)
    duration_s = end_s - start_s

    if duration_s < min_clip_duration:
        return None

    avg_confidence = sum(r["confidence"] for r in cluster) / len(cluster)

    return {
        "label": cluster[0]["label"],
        "duration_s": duration_s,
        "duration": format_mmss(duration_s),
        "starttime": format_hms(start_s),
        "endtime": format_hms(end_s),
        "start_s": start_s, # will also be used for thumbnail
        "end_s": end_s,
        "record_count": len(cluster),
        "avg_confidence": avg_confidence,
    }

# format seconds to mm:ss
def format_mmss(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"

# format seconds to hh:mm:ss
def format_hms(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_clip_thumbnail(clip: dict, video_path: str) -> list[dict]:
    return

def create_clip(clip: dict, video_path: str) -> list[dict]:
    return

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("records")
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--max-gap", type=float, default=2.0,
                        help="max seconds between detections to be considered the same clip")
    parser.add_argument("--min-records", type=int, default=2,
                        help="min number of detections required to keep a clip")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="min clip duration in seconds")
    parser.add_argument("--top-n", type=int, default=None,
                        help="only output the top N clips per label (sorted desc by --sort key)")
    parser.add_argument("--sort", type=str, default="record_count",
                        choices=["record_count", "avg_confidence", "duration_s"],
                        help="sort clips desc by this field")
    args = parser.parse_args()

    df = pd.read_csv(args.records)
    records = df.to_dict(orient="records")
    clips = parse_records_for_clips(
        records,
        fps=args.fps,
        max_gap_s=args.max_gap,
        min_records_in_clip=args.min_records,
        min_clip_duration=args.min_duration,
    )

    for label, clip_list in clips.items():
        clip_list.sort(key=lambda c: c[args.sort], reverse=True)
        if args.top_n is not None:
            clip_list = clip_list[:args.top_n]
        df = pd.DataFrame(clip_list)
        print(df)