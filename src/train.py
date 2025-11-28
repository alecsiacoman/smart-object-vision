from ultralytics import YOLO
import yaml, sys

def main(cfg_path="configs/train/yolov8n_mug.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    model = YOLO(cfg.get("model", "yolov8n.pt"))
    results = model.train(
        data=cfg["data"],
        epochs=cfg.get("epochs", 50),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        patience=cfg.get("patience", 15),
        seed=cfg.get("seed", 42),
        project="outputs",
        name="mug_yolov8n",
        exist_ok=True,
    )
    print(results)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/train/yolov8n_mug.yaml")
