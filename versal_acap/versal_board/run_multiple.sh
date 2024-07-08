#!/bin/bash

echo "Prep"
source model_src/run_all_target.sh clean_ vck190 $1

source model_src/run_all_target.sh compile_ vck190 $1

source model_src/run_all_target.sh test_images_ vck190 $1
echo "INF"
source model_src/run_all_target.sh run_cnn_ vck190 $1
source model_src/run_all_target.sh run_cnn_ vck190 $1
source model_src/run_all_target.sh run_cnn_ vck190 $1
source model_src/run_all_target.sh run_cnn_ vck190 $1
source model_src/run_all_target.sh run_cnn_ vck190 $1
source model_src/run_all_target.sh run_cnn_ vck190 $1
source model_src/run_all_target.sh run_cnn_ vck190 $1


source model_src/run_all_target.sh end_ vck190 $1
