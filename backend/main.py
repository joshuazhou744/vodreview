import pandas as pd

from detect import detect_records


def download_video():
    # TODO: implenent video download based on the object storage service i use
    pass

if __name__ == "__main__":
    video_path = "video.mp4"
    classes = ["phone", "toothbrush"]

    records = detect_records(classes, video_path, fps=4, confidence=0.1)
    records_df = pd.DataFrame(records)
    records_df.to_csv("records.csv", index=False)