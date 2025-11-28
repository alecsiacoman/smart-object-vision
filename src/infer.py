from ultralytics import YOLO
import sys, cv2, pathlib

def infer(weights, source):
    model = YOLO(weights)
    # images or video path; for webcam use source=0
    results = model.predict(source=source, imgsz=640, conf=0.25, stream=True)
    for r in results:
        frame = r.plot()
        cv2.imshow("mug-det", frame)
        if cv2.waitKey(1) & 0xFF == 27: break  # ESC to quit

if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else "outputs/mug_yolov8n/weights/best.pt"
    source  = sys.argv[2] if len(sys.argv) > 2 else 0
    infer(weights, source)
