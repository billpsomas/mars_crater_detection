import argparse
from ultralytics import YOLO

# Define the argument parser and add arguments
parser = argparse.ArgumentParser(description='YOLOv8 Training Script')
parser.add_argument('--model_name', type=str, default='yolov5x.pt', help='Model name or path')
parser.add_argument('--imgsz', type=int, default=256, help='Image size')
parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
parser.add_argument('--iou', type=float, default=0.2, help='IoU threshold')
parser.add_argument('--epochs', type=int, default=600, help='Number of epochs')
parser.add_argument('--device', nargs='+', type=int, default=[1], help='Device IDs of GPUs to use')
parser.add_argument('--batch', type=int, default=128, help='Batch size')
parser.add_argument('--data', type=str, default='data/mars.yaml', help='Path to dataset')
parser.add_argument('--name', type=str, default='yolov5x-mars-imgsz-256-batch-128', help='Name for the training session')

# Parse the arguments
args = parser.parse_args()

# Use the arguments to train the model
model = YOLO(args.model_name)
model.train(data=args.data, batch=args.batch, epochs=args.epochs, imgsz=args.imgsz, device=args.device, name=args.name)
