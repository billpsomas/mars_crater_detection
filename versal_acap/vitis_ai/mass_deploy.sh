#!/bin/bash

python vai_q_yolo.py --model_name "model_files/yolov8l.pt" --batch_size 16 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_name "model_files/yolov8l.pt" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_v8l

echo "Done with Model, Move to next?"
read

python vai_q_yolo.py  --model_name "model_files/yolov8m.pt" --batch_size 32 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_name "model_files/yolov8m.pt" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_v8m

echo "Done with Model, Move to next?"
read

python vai_q_yolo.py  --model_name "model_files/yolov8n.pt" --batch_size 32 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_name "model_files/yolov8n.pt" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_v8n

echo "Done with Model, Move to next?"
read

python vai_q_yolo.py  --model_name "model_files/yolov8s.pt" --batch_size 32 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_name "model_files/yolov8s.pt" --batch_size 1  --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_v8s

echo "Done with Model, Move to next?"
read

python vai_q_yolo.py  --model_name "model_files/yolov8x.pt" --batch_size 16 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_name "model_files/yolov8x.pt" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_v8x

echo "Done with Model, Move to next?"
read

python vai_q_yolo.py  --model_name "model_files/yolov5l.pt" --batch_size 16 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_name "model_files/yolov5l.pt" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_v5l

echo "Done with Model, Move to next?"
read

python vai_q_yolo.py  --model_name "model_files/yolov5m.pt" --batch_size 32 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_name "model_files/yolov5m.pt" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_v5m

echo "Done with Model, Move to next?"
read

python vai_q_yolo.py  --model_name "model_files/yolov5n.pt" --batch_size 32 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_name "model_files/yolov5n.pt" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_v5n

echo "Done with Model, Move to next?"
sleep 40

python vai_q_yolo.py  --model_name "model_files/yolov5s.pt" --batch_size 32 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py  --model_name "model_files/yolov5s.pt" --batch_size 32 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_name "model_files/yolov5s.pt" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_v5s

echo "Done with Model, Move to next?"
sleep 40

python vai_q_yolo.py  --model_name "model_files/yolov5x.pt" --batch_size 8 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py  --model_name "model_files/yolov5x.pt" --batch_size 8 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_name "model_files/yolov5x.pt" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_v5x

sleep 20

echo "Finished models"


