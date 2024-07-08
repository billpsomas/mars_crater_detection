#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


TARGET=$2
#vek280

MODELFILE=$3


#clean
clean_(){
echo " "
echo "cleaning"
echo " "
cd model_src
rm -rf test
rm -f *~
rm -f  run_cnn cnn* get_dpu_fps *.txt
rm -rf rpt
rm -f  *.txt
rm -f  *.log
mkdir -p rpt
cd ..
}

# compile CNN application
compile_(){
echo " "
echo "compiling binaries"
echo " "
cd model_src/code
echo "PWD1 = " $PWD
bash -x ./build_app.sh
mv code ../cnn_inference # change name of the application
bash -x ./build_get_dpu_fps.sh
mv code ../get_dpu_fps
cd ../..
echo "PWD2 = " $PWD
}

# build cifar10 test images
test_images_(){
echo " "
echo "build test images"
echo " "
cd model_src
bash ./build_testset.sh test
cd ..
echo " "
echo "PWD3 = " $PWD
}

# now run the cifar10 classification with 4 CNNs using VART C++ APIs
run_cnn_(){
echo " "
echo " run CNN"
echo " "
cd model_src
./cnn_inference ./${MODELFILE}.xmodel ./test/ ./no_labels.dat > ./rpt/predictions_${MODELFILE}.log
# check DPU prediction accuracy
#bash -x ./fps_performance.sh ${TARGET} ${MODELFILE}
echo "PWD4 = " $PWD
cd ..
}

#remove images
end_(){
echo " "
echo "end of inference"
echo " "
cd model_src
rm -rf test
cd ../
echo "PWD5 = " $PWD
#tar -cvf target.tar ./target_*
}


post_proc_(){
echo "Post Processing"
cp model_configs/${MODELFILE}_config_no_srd_reg_nc_dfl.pkl model_src/rpt/
cd ../
source apaterakis_yolo/bin/activate
cd board
python postproc.py model_src/rpt ${MODELFILE} > model_src/rpt/post_log_${MODELFILE}.log
deactivate
mv output_boxes_${MODELFILE}.txt model_src/rpt/
}

main()
{
    clean_
    compile_
    test_images_
    run_cnn_
    end_
    #post_proc_
}




"$@"
