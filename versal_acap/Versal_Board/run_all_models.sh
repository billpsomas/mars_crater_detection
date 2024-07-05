#!/bin/bash

mkdir results

source ./run_src.sh vck190 yolov5l

mkdir results/yolov5l_results
mv model_src/rpt/* results/yolov5l_results/

source ./run_src.sh vck190 yolov5m
mkdir results/yolov5m_results
mv model_src/rpt/* results/yolov5m_results/

source ./run_src.sh vck190 yolov5n
mkdir results/yolov5n_results
mv model_src/rpt/* results/yolov5n_results/


source ./run_src.sh vck190 yolov5s
mkdir results/yolov5s_results
mv model_src/rpt/* results/yolov5s_results/


source ./run_src.sh vck190 yolov5x
mkdir results/yolov5x_results
mv model_src/rpt/* results/yolov5x_results/



source ./run_src.sh vck190 yolov8l
mkdir results/yolov8l_results
mv model_src/rpt/* results/yolov8l_results/


source ./run_src.sh vck190 yolov8m
mkdir results/yolov8m_results
mv model_src/rpt/* results/yolov8m_results/


source ./run_src.sh vck190 yolov8n
mkdir results/yolov8n_results
mv model_src/rpt/* results/yolov8n_results/


source ./run_src.sh vck190 yolov8s
mkdir results/yolov8s_results
mv model_src/rpt/* results/yolov8s_results/


source ./run_src.sh vck190 yolov8x
mkdir results/yolov8x_results
mv model_src/rpt/* results/yolov8x_results/


tar -cvf yolo_results.tar results/
