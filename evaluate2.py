import argparse
from ultralytics import YOLO
import torch

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO model inference.")
    parser.add_argument('--imgsz', type=int, default=256, help='image size')
    parser.add_argument('--conf', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.2, help='IOU threshold')
    parser.add_argument('--model_name', type=str, default='/mnt/data/bill/code/yolov8/runs/detect/yolov5s-mars-imgsz-256-batch-512/weights/best.pt', help='path to model weights')
    parser.add_argument('--data', type=str, default='data/mars.yaml', help='path to data file')
    parser.add_argument('--split', type=str, default='val', help='dataset split (train/val)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Make sure your model and inputs are on the same device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a YOLOv8 model from a pre-trained weights file
    model = YOLO(args.model_name)
    model.to(device)

    # Timing inference using torch.cuda.Event
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Evaluate model - watch out it expects val.txt to evaluate on 
    start.record()
    model.val(data=args.data, imgsz=args.imgsz, conf=args.conf, iou=args.iou, split=args.split)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    inference_time = start.elapsed_time(end)  # Gives time in milliseconds

    print(f"Inference Time: {inference_time} ms")

if __name__ == "__main__":
    main()