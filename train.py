import argparse
import yaml
from ultralytics import YOLO

# Define the argument parser and add arguments
def parse_args():
    # General training arguments
    parser = argparse.ArgumentParser(description='YOLOv8 Training Script')
    parser.add_argument('--model_name', type=str, default='yolov8s.pt', help='Model name or path')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.2, help='IoU threshold')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--device', nargs='+', type=int, default=[0, 1], help='Device IDs of GPUs to use')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--data', type=str, default='/home/bill/code/yolov4/data/mbls_0_5_6_7_8_9_10_v1.yaml', help='Path to dataset')
    parser.add_argument('--name', type=str, default='yolov8s-p2-mbls-0-5-6-7-8-9-10-v1-ball-players-1280', help='Name for the training session')
    # For using feature pyramids
    parser.add_argument('--feature_pyramid', action='store_true', help='Enable additional feature pyramid levels')
    # For tuning model
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')
    parser.add_argument('--iterations', type=int, default=300, help='Maximum number of trials (hyperparameter combinations) to run')
    # Training with best hyperparameters
    parser.add_argument('--hyps', type=str, default=None, help='Best hyperparameters YAML (results from tuning) path')
    args = parser.parse_args()
    return args

def main(): 
    # Parse the arguments
    args = parse_args()

    # Use the arguments to train the model
    if args.feature_pyramid:
        model = YOLO(args.model_name.split('.')[0] + '-p2.yaml').load(args.model_name)
    else:
        model = YOLO(args.model_name)

    if args.tune:
        # Tune hyperparameters
        model.tune(data=args.data, batch=args.batch, epochs=args.epochs, iterations=args.iterations, imgsz=args.imgsz, device=args.device, name=args.name, optimizer='AdamW', plots=False, save=False, val=False)
    else:
        if args.hyps:
            # Train model using best hyperparameters 
            with open(args.hyps, 'r') as file:
                hyps = yaml.safe_load(file)        
            model.train(data=args.data, batch=args.batch, epochs=args.epochs, imgsz=args.imgsz, device=args.device, name=args.name, optimizer='AdamW', **hyps)
        else:
            # Train model using default hyperparameters 
            model.train(data=args.data, batch=args.batch, epochs=args.epochs, imgsz=args.imgsz, device=args.device, name=args.name)

if __name__ == "__main__":
    main()