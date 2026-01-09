import pandas as pd

from detect import detect_records


def download_video():
    # TODO: implenent video download based on the object storage service i use
    pass

if __name__ == "__main__":
    video_path = "vod.mp4"
    label = "dog"

    records = detect_records(label, video_path, fps=1, confidence=0.2)
    records_df = pd.DataFrame(records)
    records_df.to_csv("records.csv", index=False)