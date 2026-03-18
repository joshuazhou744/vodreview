import os
from ultralytics import YOLO


def train(dataset_yaml, epochs=50, imgsz=640, device="cpu"):
    model = YOLO("yolo26n.pt")
    
    # freeze everything except classifier head
    model.model.model[-1].requires_grad_(True)
    for param in list(model.model.parameters())[:-1]:
        param.requires_grad = False
    
    results = model.train(data=dataset_yaml, epochs=epochs, imgsz=imgsz, device=device)

    weights_path = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"[train] best weights saved to {weights_path}")
    return weights_path