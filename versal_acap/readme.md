# Versal YOLO Deployment

<ins>We will be using our Versal AI Core Series VCK190 with VitisAI 3.0</ins>

VCK190 only supports int8 quantized models along with a specific list of operations. Due to those requirements, SiLU layers (existing in all YOLO models) are unsupported along with some other operations which we will encounter later on.

To tackle those issues, we will perform some conversions on our models and their structure. For this reason, our process can be split in 4 segments:

1. [Model Layer Conversion](#models)
2. [VitisAI Quantization](#vitisai)
3. [Versal Deployment](#versal)
4. [Postprocessing and Metrics](#postprocessing)



## Model Layer Conversion

Ultralytics package is assumed installed along with PyTorch.

First and foremost, SiLU layers are fully incompatible with VitisAI so we replace them with LeakyReLU 0.1 slope layers. These layers are extremely similar to each other meaning we can also avoid retraining. However, it is worth noting, in our use case as we observe from our results, retraining would be very beneficial since both YOLOv5 and YOLOv8 use a very large number of SiLU layers.

> [!Note]
> While 0.1 slope is standard, for VitisAI we need precisely 0.1015625 slope.


To convert the SiLU layers simply run the [converter](Models/YOLO_converter.py) as:

``python YOLO_converter.py <path_to>/best.pt <output_dir> <my_model_name>``

Where:
* <path_to>/best.pt is the path to your weights file\
    (eg. "~/models/yolov5n_weights/best.pt")

* <output_dir> the path to your folder of choice where the converted model is saved to

* <model_name> the name of your final model

Example:\
``python YOLO_converter.py ../model_files/yolov8n-mars-imgsz-256-batch-512/weights/best.pt ./ yolov8n``

> [!Important]
> When models are saved from ultralytics, the weight pt files contain a dictionary with the model and its weights along other parameters. We isolate the model on which we change the SiLU layers and save it. This is done so we can use other PyTorch Functions like trace in VitisAI since it follows a very specific procedure which we cannot change.

## VitisAI

> [!Warning]
> Following conversions were done on ultralytics 8.1.47. Newer versions have altered structure and functions. DO NOT follow this guide for newer versions.


We assume you have VitisAI pre-installed and ready using a plethora of guides available including AMD's https://xilinx.github.io/Vitis-AI/3.0/html/index.html.

> [!Caution]
> VitisAI 3.0 must be used for deployment on VCK190. Other versions are unsupported.

This stage will prepare our models for Versal by quantizing and applying any required conversions. Simply copy the files of [VitisAI](VitisAI/)
to your <vitis-ai_installpath>/src/vai_quantizer/vai_q_pytorch/examples/ folder and you are ready to start.

In your VitisAI docker container first activate the pytorch environment:\
``conda activate vitis-ai-pytorch``\
and install the ultralytics package so the models can be loaded:\
``pip install ultralytics``\.

Now, before running our quantization and compilation scripts, 2 major changes must be done to the model class definitions.

It would be possible to just include the required model class files however, there are many dependencies so we found it easier to temporarily change the package code.

It is also worth noting that with our previous SiLU conversion, we no longer use the ultralytics YOLO class but the DetectionModel backend class. This has no effect on the performance, it simply means we lose some of the automated functions.


Step 1: Locate where the ultralytics package is installed.\
It should be either in the global python packages or in user packages.
Check the logs from pip install. 

Step 2: Navigate to the ultralytics package folder and locate the
``ultralytics/nn/models/`` folder.

Here open *block.py* with your terminal text editor of choice.

Step 3: Change the DFL forward return function as follows:

Previous DFL Class Forward function:
```
    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
```


New DFL class forward function:
```
def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(torch.cat([x.view(b,4,self.c1,a).split(1,dim=2)[i].squeeze(2).unsqueeze(1) for i in range(x.view(b,4,self.c1,a).size(2))], dim=1).softmax(1)).view(b,4,a)
```

> [!Note]
> Our return command change does not affect the model or its performance, it produces the same exact result but also fixes VitisAI implications with the transpose function

Step 4: Save your changes 

Step 5: Locate in the same folder *head.py* file and open it.\
(Path: ultralytics/nn/modules/head.py)

Here we will be modifying the forward function of our Detect class.

Old forward function:
```
def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
```


New forward function:

```
def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.shape = shape
        return x
```

The removed functionality will be added back on our postprocessing stage since it can't run on our board.
In order however to be able to run them we will have to extract some values as a configuration file for later use. 

Step 6: This will be done along quantization in [vai_q_yolo.py](VitisAI/vai_q_yolo.py).

You can run the script on its own with the following arguments:

``python vai_q_yolo.py  --model_name <path_to_your_model.pt> --batch_size <batch_size> --target DPUCVDX8G_ISA3_C32B6 --quant_mode <calib|test>``

Where:
* model_name should be the converted pt file from Model_Conversion segment (eg. model_files/yolov5n.pt)

* batch_size your batch_size. Here use a small size of 32 or 16 since RAM requirements are increased and most likely your program will get killed.

* quant_mode either calibration or test. You always have to run calibration before testing. Test mode also is responsible for deployment of the model for Versal.

We recommend to running our script [mass_deploy.sh](VitisAI/mass_deploy.sh) which calibrates and deploys every model and their configuration files.

> [!Note]
> If you want to run it as is: 1. extract the val.tar in the VitisAI/data/ folder. 2. Check the calibration stage, if it fails for any reason simply re-run it.

Having quantized the models, they require a final conversion handled by VitisAI in order to produce the final DPU subgraph for Versal. 

Step 7: Run ``source /workspace/board_setup/vck190/host_cross_compiler_setup.sh``

Step 8: Run ``unset LD_LIBRARY_PATH`` and
``source $install_path/environment-setup-cortexa72-cortexa53-xilinx-linux``.

Where $install_path should be provided from the terminal output as the whole command.

Step 9: Now you can compile the extracted models using vai_c_xir as follows:
``vai_c_xir -x <Vitis_AI_export_folder/DetectionModel_int.xmodel> -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o <output_folder> -n <output_model_name>``

Example: ``vai_c_xir -x quantize_result_v5m/DetectionModel_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o versal_output/yolov5m/ -n yolov5m``

Here you can also run our [mass_compile.sh](VitisAI/mass_compile.sh) script to compile all models.



## Versal

Now the models are finished, we copy [Versal_Board](Versal_Board/) folder on our Versal VCK190 board and execute [run_src.sh](Versal_Board/run_src.sh) script as follows:

``./run_src vck190 <model_filename>``

Or we can run [run_all_models.sh](Versal_Board/run_all_models.sh) to automatically run all our models and compress the results.

> [!Note]
> Extract models.zip.tar first to access the finalized models
## Postprocessing

Retrieving our results, we can run postprocessing using our configuration files. 

Step 1: Generate the box predictions from Versal raw output.
``python postproc.py <path_to_raw_output> <model_name> <iou_threshold>`` 

Example: python postproc.py ~/Versal_results/yolov5n_results/ yolov5n 0.1

Where: 
* <path_to_raw_output> should be the folder with the config file and the versal output log

* <model_name> name contained in both files (eg. yolov5n)

* <iou_threshold> Prefered IOU. Conf threshold is pre-set to 0.05.


This will create a new file ``output_boxes_<model_name>.txt`` with all box predictions which needs conversion to coco format.

Step 2: Convert to desired format with [coco-prep.py](Postprocessing/coco-prep.py)

``python coco-prep.py <model_name>``.

This will generate a new file "coco-preped-<model_name>.json" for coco-evaluation.

Finally,\
Step 3: Evaluate results with [evaluate.py](Postprocessing/evaluate.py)

``python evaluate.py coco-preped-<model_name>.json``

