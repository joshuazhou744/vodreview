import pandas as pd

def parse_records_for_clips(
        records: list[dict],
        fps: float,
        max_gap_s: float = 2.0,
        min_clip_duration: float = 0.5,
        max_clip_duration: float = 20.0,
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
        # no detections for this label
        if not items: continue
        # sort records of each label by time
        items.sort(key=lambda r: r["timestamp_s"])
        # store the clips that we find
        label_clips = []

        # iterator for item array
        it = iter(items)
        # set prev to first item and iterate to second item
        prev = next(it)
        
        # start the current cluster of records (building candidate clip) with the first record
        current_cluster = [prev]
        # iterate over the iterator for each record (item) for the label
        for item in it:
            if item["timestamp_s"] - prev["timestamp_s"] <= max_gap_s:
                current_cluster.append(item)
            else:
                if check_cluster(current_cluster, min_clip_duration=min_clip_duration, min_records_in_clip=min_records_in_clip, fps=fps):
                    label_clips.append(cluster_to_clip(current_cluster, fps))
                current_cluster = [item]
            # item get iterated to next but prev stays the current item
            prev = item

        # check for leftover cluster
        if check_cluster(current_cluster, min_clip_duration=min_clip_duration, min_records_in_clip=min_records_in_clip, fps=fps):
            label_clips.append(cluster_to_clip(current_cluster, fps))
        
        clips[label] = label_clips
    return clips

def cluster_to_clip(cluster, fps):
    start_s = cluster[0]["timestamp_s"]
    end_s = cluster[-1]["timestamp_s"] + (1.0 / fps)
    duration_s = end_s - start_s
    avg_confidence = sum(r["confidence"] for r in cluster) / len(cluster)
    
    return {
        "label": cluster[0]["label"],
        "duration_s": duration_s,
        "duration": format_mmss(duration_s),
        "starttime": format_mmss(start_s),
        "start_s": start_s, # will also be used for thumbnail
        "end_s": end_s,
        "record_count": len(cluster),
        "avg_confidence": avg_confidence,
    }

def check_cluster(cluster, min_clip_duration, min_records_in_clip, fps):
    if not cluster:
        return False
    duration_s = cluster[-1]["timestamp_s"] - cluster[0]["timestamp_s"] + (1.0 / fps)
    return duration_s >= min_clip_duration and len(cluster) >= min_records_in_clip

# format seconds to mm:ss
def format_mmss(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"

def get_clip_thumbnail(clip: dict, video_path: str) -> list[dict]:
    return

def create_clip(clip: dict, video_path: str) -> list[dict]:
    return

if __name__ == "__main__":
    df = pd.read_csv("records.csv")
    records = df.to_dict(orient="records")
    clips = parse_records_for_clips(
        records,
        fps=4,
    )


    for label, clip_list in clips.items():
        df = pd.DataFrame(clip_list)
        print(df)