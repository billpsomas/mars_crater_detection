#!/bin/bash

vai_c_xir -x quantize_result_v5m/DetectionModel_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o versal_output/yolov5m/ -n yolov5m
sleep 5

vai_c_xir -x quantize_result_v5s/DetectionModel_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o versal_output/yolobv5s/ -n yolov5s
sleep 5

vai_c_xir -x quantize_result_v5l/DetectionModel_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o versal_output/yolov5l/ -n yolov5l
sleep 5

vai_c_xir -x quantize_result_v5n/DetectionModel_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o versal_output/yolov5n/ -n yolov5n
sleep 5

vai_c_xir -x quantize_result_v5x/DetectionModel_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o versal_output/yolov5x/ -n yolov5x
sleep 5

vai_c_xir -x quantize_result_v8m/DetectionModel_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o versal_output/yolov8m/ -n yolov8m
sleep 5

vai_c_xir -x quantize_result_v8s/DetectionModel_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o versal_output/yolobv8s/ -n yolov8s
sleep 5

vai_c_xir -x quantize_result_v8l/DetectionModel_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o versal_output/yolov8l/ -n yolov8l
sleep 5

vai_c_xir -x quantize_result_v8n/DetectionModel_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o versal_output/yolov8n/ -n yolov8n
sleep 5

vai_c_xir -x quantize_result_v8x/DetectionModel_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o versal_output/yolov8x/ -n yolov8x
sleep 5