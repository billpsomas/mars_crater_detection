import argparse
import json
import os
from PIL import Image
from ultralytics import YOLO

def convert_bboxes(bxs):
    converted_bboxes = []
    for bx in bxs:
        x_center, y_center, width, height = bx
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        converted_bboxes.append([xmin, ymin, width, height])
    return converted_bboxes

# Argument parser
parser = argparse.ArgumentParser(description='YOLOv8 Model Evaluation')
parser.add_argument('--imgsz', type=int, default=256, help='image size')
parser.add_argument('--conf', type=float, default=0.05, help='confidence threshold')
parser.add_argument('--iou', type=float, default=0.5, help='IOU threshold')
parser.add_argument('--source', type=str, default="/mnt/data/datasets/2022_GeoAI_Martian_Challenge_Dataset/val_images", help='source directory for images')
parser.add_argument('--save_dir', type=str, default="/mnt/data/bill/code/yolov8/runs/detect/yolov5x-mars-256-evaluation-results/", help='directory to save results')
parser.add_argument('--json_name', type=str, default="results.json", help='name of the JSON file to save results')
parser.add_argument('--save_imgs', action='store_true', help='flag to save images')
parser.add_argument('--save_txts', action='store_true', help='flag to save text files')
parser.add_argument('--model_name', type=str, default="/mnt/data/bill/code/yolov8/runs/detect/yolov5x-mars-imgsz-256-batch-128/weights/best.pt", help='path to the model weights')

args = parser.parse_args()

# Use the arguments
os.makedirs(args.save_dir, exist_ok=True)

if args.save_imgs:
    os.makedirs(os.path.join(args.save_dir, 'imgs'), exist_ok=True)

if args.save_txts:
    os.makedirs(os.path.join(args.save_dir, 'txts'), exist_ok=True)

# Load a YOLOv8 model from a pre-trained weights file
model = YOLO(args.model_name)

results = model(args.source, classes=0, imgsz=args.imgsz, conf=args.conf, iou=args.iou, stream=True)

# Initialize an empty dictionary or load existing data
json_file_path = os.path.join(args.save_dir, args.json_name)
if os.path.exists(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
else:
    data = {}

for res in results:
    im_id = res.path.split('/')[-1].split('.')[0] # img id

    if args.save_imgs:
        im_array = res.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save(os.path.join(args.save_dir, im_id + '.png'))  # save image

    if args.save_txts:
        res.save_txt(os.path.join(args.save_dir, res.path.split('/')[-1].replace('.png', '.txt'))) # save txt 

    # Save JSON for official COCO format evaluation
    bxs = res.boxes.xywh.cpu().detach().numpy().tolist() # export boxes to a list
    bxs = convert_bboxes(bxs)
    cnfs = res.boxes.conf.cpu().detach().numpy().tolist() # export respective confidences to a list

    merged = [sublist + [other] for sublist, other in zip(bxs, cnfs)] # merge
    rounded = [[round(item, 5) for item in sublist] for sublist in merged] # round

    data[im_id] = rounded # append the data to the dictionary

    # Write the updated dictionary to the file
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)