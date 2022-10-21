#!/bin/bash

dir="sampler"

multigpu() {
    python3 $dir/multigpu.py --gpus $1 >logs/$1gpu.log
    pid=$!
    wait $pid
    echo "$@"
}

test() {
    multigpu 4
    multigpu 2
    multigpu 1
    # epochtimetest 4
    # epochtimetest 2
    # epochtimetest 1
}

compare() {
    python $dir/gpu1_microbatch.py --num_epochs $1 >logs/gpu1_microbatch.log
    pid=$!
    wait $pid
    echo "$@"
    python $dir/gpu1_minibatch.py --num_epochs $1 >logs/gpu1_minibatch.log
}

epochtimetest() {
    python3 $dir/epochtimetest.py --gpus $1 >logs/epochtimetest$1gpu.log
    pid=$!
    wait $pid
    echo "$@"
}

clear() {
    rm logs/*gpu.log
}

compare 4
