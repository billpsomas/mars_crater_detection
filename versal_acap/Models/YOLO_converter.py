# SiLU Layers are unsupported so we change them to LeakyReLU with 0.1 slope
# Ideally you would want to retrain the model, thats up to you.

import ultralytics
import torch
import sys
import os

def model_conversion(model):
    def _iter(module, prefix=''):
        for name, sub_module in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(sub_module, torch.nn.SiLU) and sub_module.__class__.__name__ == "SiLU":
                #Versal requires 0.1015625 slope not 0.1
                setattr(module, name, torch.nn.LeakyReLU(0.1015625))
            _iter(sub_module, prefix=full)
    _iter(model)


if __name__ == "__main__":
    if len(sys.argv) != 4: 
        print("Use: python YOLO_converter.py <path_to>/best.pt <output_dir> <my_model_name>")
    else:
        #Load YOLO weights
        model = torch.load(sys.argv[1])
        #From the dict select the model (DetectionModel)
        model = model["model"]
        model = model.float().to("cpu")

        #REMOVE IF YOU INTEND TO RETRAIN
        model.eval()

        #Convert SiLU to LeakyReLU
        model_conversion(model)
        torch.save(model, os.path.join(sys.argv[2], f"{sys.argv[3]}.pt"))