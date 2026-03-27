from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import glob
import yaml

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)


def auto_label(image_dir, label, output_dir=None, **kwargs):
    if output_dir is None:
        output_dir = os.path.dirname(image_dir)

    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(image_dir, "*")))
    labeled = 0

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        prompt = f"<OPEN_VOCABULARY_DETECTION>{label}"
        inputs = processor(text=prompt, images=img, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=1024, num_beams=1)
        result = processor.batch_decode(outputs, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(result, task="<OPEN_VOCABULARY_DETECTION>", image_size=(w, h))

        bboxes = parsed.get("<OPEN_VOCABULARY_DETECTION>", {}).get("bboxes", [])
        if not bboxes:
            continue

        txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_name)
        with open(txt_path, "w") as f:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        labeled += 1

    purged = 0
    for img_path in images:
        txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        if not os.path.exists(os.path.join(labels_dir, txt_name)):
            os.remove(img_path)
            purged += 1

    print(f"[label] labeled {labeled}/{len(images)} images, purged {purged}")

    # make dataset.yaml spec
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    dataset_config = {
        "path": os.path.abspath(output_dir),
        "train": "images",
        "val": "images",
        "names": {0: label},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f)

    print(f"[label] dataset.yaml written to {yaml_path}")
    return yaml_path