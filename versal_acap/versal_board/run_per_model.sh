#!/bin/bash

MODELFILE = $1


source ./run_src.sh vck190 $1

mkdir results/$1_results
mv model_src/rpt/* results/$1_results/

