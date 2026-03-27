import pandas as pd
import os

from scrape import scrape_images
from train import train
from detect import detect_records_yolo

os.environ["TOKENIZERS_PARALLELISM"] = "false" # suppress parallelism warning between huggingface and ultralytics

if __name__ == "__main__":
    import argparse
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument("prompts")
    parser.add_argument("video_path")
    parser.add_argument("--label", type=str, default=None, help="class label (defaults to prompt)")
    parser.add_argument("--labeler", type=str, default="yoloworld", choices=["yoloworld", "florence"])
    parser.add_argument("--clean", action="store_true", help="delete dataset after training")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--label-confidence", type=float, default=0.5, help="YOLO-World labeling threshold")
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.labeler == "florence":
        from label_florence import auto_label
    else:
        from label import auto_label

    # scrape
    images = scrape_images(args.prompts, max_images=args.max_images)
    if not images:
        print(f"[main] no images scraped for '{args.prompts}', aborting")
        exit(1)

    # label
    label = args.label or args.prompts
    dataset_yaml = auto_label(os.path.dirname(images[0]), label, confidence_threshold=args.label_confidence)

    # train
    weights = train(dataset_yaml, epochs=args.epochs, device=args.device)

    # detect
    records = detect_records_yolo(label, args.video_path, weights, fps=args.fps, confidence=args.confidence)

    # save
    records_df = pd.DataFrame(records)
    records_df.to_csv(f"{args.prompts}_records.csv", index=False)
    print(f"[main] saved {len(records)} detections to {args.prompts}_records.csv")

    if args.clean:
        dataset_dir = os.path.dirname(dataset_yaml)
        shutil.rmtree(dataset_dir)
        print(f"[main] cleaned up {dataset_dir}")