# Evaluation of Resource-Efficient Crater Detectors on Embedded Systems
This is the accompanying repository to the following IGARSS 2024 oral paper: [[arXiv](https://arxiv.org/abs/2405.16953), [slides](http://users.ntua.gr/psomasbill/Mars_IGARSS2024_slides.pdf)]

## Overview
Real-time analysis of Martian craters is crucial for mission-critical operations, including safe landings and geological exploration. This work leverages the latest breakthroughs for on-the-edge crater detection aboard spacecraft. We rigorously benchmark several YOLO networks using a Mars craters dataset, analyzing their performance on embedded systems with a focus on optimization for low-power devices. We optimize this process for a new wave of cost-effective, commercial-off-the-shelf-based smaller satellites. Implementations on diverse platforms, including Google Coral Edge TPU, AMD Versal SoC VCK190, Nvidia Jetson Nano and Jetson AGX Orin, undergo a detailed trade-off analysis. Our findings identify optimal network-device pairings, enhancing the feasibility of crater detection on resource-constrained hardware and setting a new precedent for efficient and resilient extraterrestrial imaging.

## Objectives
1. Train several efficient crater detection networks on a publicly available benchmark dataset
2. Evaluate these networks for their model and device performance on selected embedded devices
3. Conduct a network/device pair exploration and trade-off analysis



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
python3 predict.py --model_name 'yolov8n-mars-256/weights/best.pt' --source 'val_images' --save_dir 'yolov8n-mars-256-evaluation-results/' --json_name 'results.json' --imgsz 256 --conf 0.4 --data 'data/mars.yaml'
```

This will solely create the JSON file needed for official evaluation. 
If you want to save images(txts) with predicted boxes use `--save_imgs`(`--save_txts`) respectively.


# Evaluate models
In order to evaluate the models following the official protocol, we will need the `gt_eval.json` file, which serves as the groundtruth. The `results.json` produced through `predict.py` has our model's predictions. We compare these two.

## YOLOv8 nano model 
```bash
python3 evaluate.py 'results.json'
```

WATCH OUT: `gt_eval.json` has only train and val groundtruths (we currently use val). In order to evaluate on test set, we need to register to the official competition. 

---
# Deploy the Models
In order to facilitate the reproduction of the experiments, we have made all necessary model files available to the user for all device architectures presented in the relevant paper. For all model variations, the original pytorch (`.pt`) model was used as the reference model. Having created the pytorch model in each case, we exported all other model formats required, using the Ultralytics library export function.
We also include the ONNX-format for all models to enable greater flexibility in future re-use of the work in this repository.

In detail, the following model files are available for each processing architecture:
### CPU Deployment
All CPU experiments were conducted using a FP32-format pytorch (`.pt`) model file for each network. The same model file was used both for the high-end Desktop CPU and the embedded CPU. 

### GPU Deployment
For the high-end, server GPU, a half-precision FP16 pytorch (`.pt`) model was used. This was dervived from the original FP32 pytorch model.

### Jetson Deployment (Embedded GPU)
In the case of the Nvidia Jetson devices (Jetson Nano and Jetson Orin), the FP32 pytorch models were converted to FP32 TensorRT format (`.engine`), optimized for Jetson GPUs. This was done as TensorRT allows for maximum performance on the Nvidia devices. Alternatively, deployment on the Jetson GPUs can also be achieved using the pytorch (`.pt`) model format, sacrificing some performance.

### Google Coral Deployment (Edge TPU)
The Edge TPU supports only TensorFlow Lite models that are fully 8-bit quantized and then compiled specifically for the Edge TPU using the Google Edge TPU compiler. The above process can either be done manually or by using the Ultralytics export function. We include the Edge TPU converted `.tflite` model files in this repository.

### AMD Versal Deployment (ACAP)
Check out the [Versal README](./versal_acap/readme.md) for details on deploying YOLO on the AMD Versal.

## Results
Deploying the above models on the presented embedded devices, will give you the results presented in Table 1 of our paper

## Citation
The following BibTeX can be used to cite this work.
```
@misc{vellas2024evaluationresourceefficientcraterdetectors,
      title={Evaluation of Resource-Efficient Crater Detectors on Embedded Systems}, 
      author={Simon Vellas and Bill Psomas and Kalliopi Karadima and Dimitrios Danopoulos and Alexandros Paterakis and George Lentaris and Dimitrios Soudris and Konstantinos Karantzalos},
      year={2024},
      eprint={2405.16953},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.16953}, 
}
```


