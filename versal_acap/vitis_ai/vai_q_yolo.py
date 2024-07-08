import os
import re
import sys
import argparse
import time
import pdb
import random

from pytorch_nndct.apis import torch_quantizer
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import json
import cv2
import numpy as np
from PIL import Image


import glob
import math
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
import traceback
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_name',
    default="yolov8n",
    help='Model Variant to process. Eg: yolov8x'
)

parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')

parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')

parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')


parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')

parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')

parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')

parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')

parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')

args, _ = parser.parse_known_args()


def run_model_exports(model):
    import pickle
    tensor_no = model.model[-1].no
    tensor_stride = model.model[-1].stride
    tensor_reg_max = model.model[-1].reg_max
    tensor_nc = model.model[-1].nc
    layer_dfl = model.model[-1].dfl
    tensor_shape = model.model[-1].shape
    with open("quantize_result/{}_config_no_srd_reg_nc_dfl.pkl".format(args.model_name[12:-3]),'wb') as f:
        pickle.dump((tensor_no,tensor_stride,tensor_reg_max,tensor_nc,layer_dfl), f)
    return


IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"

class LoadImagesAndVideos:
    """
    YOLOv8 image/video dataloader.

    This class manages the loading and pre-processing of image and video data for YOLOv8. It supports loading from
    various formats, including single image files, video files, and lists of image and video paths.

    Attributes:
        files (list): List of image and video file paths.
        nf (int): Total number of files (images and videos).
        video_flag (list): Flags indicating whether a file is a video (True) or an image (False).
        mode (str): Current mode, 'image' or 'video'.
        vid_stride (int): Stride for video frame-rate, defaults to 1.
        bs (int): Batch size, set to 1 for this class.
        cap (cv2.VideoCapture): Video capture object for OpenCV.
        frame (int): Frame counter for video.
        frames (int): Total number of frames in the video.
        count (int): Counter for iteration, initialized at 0 during `__iter__()`.

    Methods:
        _new_video(path): Create a new cv2.VideoCapture object for a given video path.
    """

    def __init__(self, path, batch=1, vid_stride=1):
        """Initialize the Dataloader and raise FileNotFoundError if file not found."""
        parent = None
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()  # list of sources
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())  # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))  # glob
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))  # dir
            elif os.path.isfile(a):
                files.append(a)  # files (absolute or relative to CWD)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))  # files (relative to *.txt file parent)
            else:
                raise FileNotFoundError(f"{p} does not exist")

        # Define files as images or videos
        images, videos = [], []
        for f in files:
            suffix = f.split(".")[-1].lower()  # Get file extension without the dot and lowercase
            if suffix in IMG_FORMATS:
                images.append(f)
            elif suffix in VID_FORMATS:
                videos.append(f)
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.ni = ni  # number of images
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = batch
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {p}. {FORMATS_HELP_MSG}")

    def __iter__(self):
        """Returns an iterator object for VideoStream or ImageFolder."""
        self.count = 0
        return self

    def __next__(self):
        """Returns the next batch of images or video frames along with their paths and metadata."""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:  # end of file list
                if len(imgs) > 0:
                    return paths, imgs, info  # return last partial batch
                else:
                    raise StopIteration

            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = "video"
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)

                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break  # end of video or failure

                if success:
                    success, im0 = self.cap.retrieve()
                    if success:
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                        if self.frame == self.frames:  # end of video
                            self.count += 1
                            self.cap.release()
                else:
                    # Move to the next file if the current video ended or failed to open
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            else:
                self.mode = "image"
                im0 = cv2.imread(path)  # BGR
                if im0 is None:
                    raise FileNotFoundError(f"Image Not Found {path}")
                paths.append(path)
                imgs.append(im0)
                info.append(f"image {self.count + 1}/{self.nf} {path}: ")
                self.count += 1  # move to the next file
                if self.count >= self.ni:  # end of image list
                    break

        return paths, imgs, info

    def _new_video(self, path):
        """Creates a new video capture object for the given path."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        """Returns the number of batches in the object."""
        return math.ceil(self.nf / self.bs)  # number of files

def experimental(model):
    dataset = LoadImagesAndVideos("data/val_ids.txt", args.batch_size)
    for frame_idx, batch in tqdm(enumerate(dataset),total = 1000/args.batch_size):
        # processing
        path, transform_im, s = batch
        batch_tensor = torch.stack([torch.from_numpy(data.transpose(2,0,1)).to("cpu").float() / 255.0 for data in transform_im])
        batch_tensor = batch_tensor.float()

        temp = model(batch_tensor)
    print("Done")

def quantization(title='optimize',
                 model_name='', 
                ): 
    print("quantization")
    quant_mode = args.quant_mode
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    config_file = args.config_file
    target = args.target

    model = torch.load(args.model_name, map_location=torch.device("cpu"))
    model.float()
    model.eval()

    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1 or subset_len != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        subset_len = 1

    input = torch.randn([batch_size, 3, 256, 256])

    ####################################################################################
    # This function call will create a quantizer object and setup it. 
    # Eager mode model code will be converted to graph model. 
    # Quantization is not done here if it needs calibration.
    quantizer = torch_quantizer(quant_mode, model, (input), device=device, quant_config_file=config_file, target=target)
    print("quantizer done")

    if quant_mode == 'calib':
        print("Exporting model configuration...")
        run_model_exports(model)
        print("Done exporting configuration")
    # Get the converted model to be quantized.
    quant_model = quantizer.quant_model
    #####################################################################################

    # This function call is to do forward loop for model to be quantized.
    # Quantization calibration will be done after it if quant_mode is 'calib'.
    # Quantization test  will be done after it if quant_mode is 'test'.
    proc_pred = experimental(quant_model)

    # handle quantization result
    if quant_mode == 'calib':
    # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel()


if __name__ == '__main__':
  import sys

  file_path = os.path.join(args.model_name)
  feature_test = ' quantization'
  # force to merge BN with CONV for better quantization accuracy
  args.optimize = 1
  feature_test += ' with optimization'
  
  title = args.model_name + feature_test

  print("-------- Start {} test ".format(args.model_name))

  # calibration or evaluation
  quantization(
      title=title,
      model_name=args.model_name)

  print("-------- End of {} test ".format(args.model_name))
