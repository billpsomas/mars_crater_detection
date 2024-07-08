"""
This script reads the prediction file dump. After each printed Image name follows 3 lines containing the tensor values in shape (32,32,65).

Read the file and append each line in a list separating the numbers based on spaces.

Track the Image name line and save the following 3 indexes for the tensors.

Assign the values in the tensors and transpose to get finalized (65,32,32) shape.
Wrap all tensors with a singular tensor: (1,65,32,32), (1,65,16,16), (1,65,8,8)

Return in list type same as x before proceeding to the config postprocessing.
"""

import pickle
import ultralytics
import torch
import numpy as np
from ultralytics.utils.ops import non_max_suppression, scale_boxes
import re
import os
import sys
import time


def parse_version(version="0.0.0") -> tuple:
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        print(f"WARNING !! failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0
    
def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    from importlib import metadata
    if not current:  # if current is '' or None
        print(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError((f"WARNING !! {current} package is required but not installed")) from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op in {">=", ""} and not (c >= v):  # if no constraint passed assume '>=required'
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"WARNING !! {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(warning)  # assert version requirements met
        if verbose:
            print(warning)
    return result

TORCH_1_10 = check_version(torch.__version__, "1.10.0")

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def decode_bboxes(bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


def run_model_config(x, pth, name):
    shape = x[0].shape
    with open(os.path.join(pth, "{}_config_no_srd_reg_nc_dfl.pkl".format(name)),'rb') as f:
        tensor_no ,tensor_stride, tensor_reg_max, tensor_nc, layer_dfl = pickle.load(f)

    x_cat = torch.cat([xi.view(shape[0], tensor_no, -1) for xi in x], 2)
    tensor_anchors, tensor_strides = (x.transpose(0, 1) for x in make_anchors(x, tensor_stride, 0.5))
    box, cls = x_cat.split((tensor_reg_max * 4, tensor_nc), 1)
    dbox = decode_bboxes(layer_dfl(box),tensor_anchors.unsqueeze(0)) * tensor_strides

    y = torch.cat((dbox, cls.sigmoid()), 1)
    return (y, x)


def res_exp(pred, pth, name,iou):
    global f
    start = time.time()

    predi = run_model_config(pred,pth,name)

    box_conf = non_max_suppression(prediction=predi, conf_thres= 0.05, iou_thres= iou)
    box_conf[0] = box_conf[0].detach().numpy()
    box_conf[0] = np.delete(box_conf[0], 5, axis=1)

    box_conf[0][box_conf[0]<0] = 0

    #box_vals = box_conf[0][:, :-1]
    #confi_vals = box_conf[0][:, -1].reshape(-1, 1)
    fixed = []
    for i in box_conf[0]:
        fixed.append([i[0], i[1], i[2]-i[0], i[3]-i[1], i[4]])
    
    end = time.time() - start

    global timing
    timing += end

    f.write("Boxes:\n")

    f.write("\n")

    f.write("\n".join(" ".join(map(str, x)) for x in np.array(fixed)))

    f.write("\n")

    f.write("\n")

def box_processing_writing(dictionary, pth, name,iou):
    global f
    for named, tensor in zip(dictionary["names"], dictionary["tensors"]):
        f.write(f"Image Prediction: {named}\n")
        f.write("\n")
        res_exp(tensor,pth,name,iou)

def convert_to_tensor(pairs):
    print("Fixing Tensors")
    start = time.time()

    new_pairs = []
    for tensor_pair in pairs:
        tensor_1 = tensor_pair[0]

        tensor_2 = tensor_pair[1]

        tensor_3 = tensor_pair[2]

        for index, i in enumerate(tensor_1):
            tensor_1[index] = float(i)

        for index, i in enumerate(tensor_2):
            tensor_2[index] = float(i)

        for index, i in enumerate(tensor_3):
            tensor_3[index] = float(i)
        
        assert len(tensor_1) == 32 *  32 * 65
        assert len(tensor_2) == 16 *  16 * 65
        assert len(tensor_3) == 8 *  8 * 65

        tensor_1 = torch.tensor(tensor_1).reshape(32, 32, 65).unsqueeze(0).permute(0,3,1,2)
        tensor_2 = torch.tensor(tensor_2).reshape(16, 16, 65).unsqueeze(0).permute(0,3,1,2)
        tensor_3 = torch.tensor(tensor_3).reshape(8, 8, 65).unsqueeze(0).permute(0,3,1,2)
        new_pairs.append((tensor_1,tensor_2,tensor_3))


    end = time.time() - start
    print(f"Finished Tensors. Time: {end}s Average time per Image: {end/len(pairs)}s Average time per tensor per image: {(end/len(pairs))/3}s ")
    return new_pairs

def tensor_fix(pth,model_name):
    import os
    lines_list = []


    tensor_index = {
        "names": [],
        "tensors": []
    }

    print("Processing File")
    start = time.time()
    # Open the file in read mode ('r')
    with open(os.path.join(pth,'predictions_{}.log'.format(model_name)), 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Split the line into items separated by spaces
            items = line.strip().split(" ")

            # Append the list of items to lines_list
            lines_list.append(items)

    for line in lines_list:
        if line[0] == "Image":
            image_tensor_name = line[2]
            tensor_pair = (lines_list[lines_list.index(line) + 1] , lines_list[lines_list.index(line) + 2],lines_list[lines_list.index(line) + 3])

            tensor_index["names"].append(image_tensor_name)
            tensor_index["tensors"].append(tensor_pair)

    del lines_list
    print(f"Finished. Time: {(time.time() - start)}s")
    tensor_index["tensors"] = convert_to_tensor(tensor_index["tensors"])

    return tensor_index

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Run as: python postproc.py <path_to_raw_output> <model_name> <iou_threshold>. Eg. python3 postproc.py ~/Versal_results/yolov5n_results/ yolov5n 0.1")
    else:
        global timing
        timing = 0
        pth = sys.argv[1]
        name = sys.argv[2]
        iou = sys.argv[3]
        global f
        f = open("output_boxes_{}.txt".format(name), 'w')

        x_dict = tensor_fix(pth,name)

        box_processing_writing(x_dict,pth,name,iou)
        print(f'Total time for image post-processing: {timing}s Average time per image post-procesing: {timing/len(x_dict["names"])}s')
