#!/bin/bash


test(){
    python3 $1/multigpu.py --gpus 4 > logs/4gpu.log
    pid=$!
    wait $pid
    python3 $1/multigpu.py --gpus 2 > logs/2gpu.log
    pid=$!
    wait $pid
    python3 $1/singlegpu.py > logs/singlegpu.log
}

test $1



