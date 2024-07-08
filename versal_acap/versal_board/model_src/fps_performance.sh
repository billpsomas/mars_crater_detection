#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


TARGET=$1

MODELFILE=$2

# check DPU prediction top1_accuracy
echo " "
echo " ${MODELFILE} FPS"
echo " "


echo "THREAD 1"
./get_dpu_fps ./${MODELFILE}.xmodel  1 1000  >  ./rpt/log1.txt  # 1 thread
echo "THREAD 2"
./get_dpu_fps ./${MODELFILE}.xmodel  2 1000  >  ./rpt/log2.txt  # 2 threads
echo "THREAD 3"
./get_dpu_fps ./${MODELFILE}.xmodel  3 1000  >  ./rpt/log3.txt  # 3 threads
echo "THREAD 4"
./get_dpu_fps ./${MODELFILE}.xmodel  4 1000  >  ./rpt/log4.txt  # 4 thread
echo "THREAD 5"
./get_dpu_fps ./${MODELFILE}.xmodel  5 1000  >  ./rpt/log5.txt  # 5 thread
echo "THREAD 6"
./get_dpu_fps ./${MODELFILE}.xmodel  6 1000  >  ./rpt/log6.txt  # 6 thread


cat ./rpt/log1.txt ./rpt/log2.txt ./rpt/log3.txt ./rpt/log4.txt ./rpt/log5.txt ./rpt/log6.txt >  ./rpt/${MODELFILE}_results_fps.log
rm -f ./rpt/log?.txt

echo " "
