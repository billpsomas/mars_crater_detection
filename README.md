# Train models

## YOLOv8 nano model 
```bash
python3 train.py  --model_name 'yolov8n.pt' --imgsz 256 --conf 0.4 --iou 0.2 --epochs 600 --batch 512 --data 'data/mars.yaml' --name 'yolov8n-mars-256'
```

## YOLOv8 small model 
```bash
python3 train.py  --model_name 'yolov8s.pt' --imgsz 256 --conf 0.4 --iou 0.2 --epochs 600 --batch 512 --data 'data/mars.yaml' --name 'yolov8s-mars-256'
```

## YOLOv8 medium model 
```bash
python3 train.py  --model_name 'yolov8m.pt' --imgsz 256 --conf 0.4 --iou 0.2 --epochs 600 --batch 256 --data 'data/mars.yaml' --name 'yolov8m-mars-256'
```

# Predict models on [GeoAI Martial Challenge](http://cici.lab.asu.edu/martian/#data-dataset) dataset

## YOLOv8 nano model 
```bash
python3 predict.py --model_name 'yolov8n-mars-256/weights/best.pt' --source 'val_images' --save_dir 'yolov8n-mars-256-evaluation-results/' --json_name 'results.json' --imgsz 256 --conf 0.4 --data data/mars.yaml
```

This will solely create the JSON file needed for official evaluation. 
If you want to save images(txts) with predicted boxes use `--save_imgs`(`--save_txts`) respectively.

# Evaluate models