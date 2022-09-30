#!/bin/bash

dir=$1




multigpu(){
    python3 $dir/multigpu.py --gpus $1 > logs/$1gpu.log
    pid=$!
    wait $pid
    echo "$@"
}


test(){
    multigpu 4
    multigpu 2
    multigpu 1
    # epochtimetest 4
    # epochtimetest 2
    # epochtimetest 1
}



epochtimetest(){
    python3 $dir/epochtimetest.py --gpus $1 > logs/epochtimetest$1gpu.log
    pid=$!
    wait $pid
    echo "$@"
}

test 



