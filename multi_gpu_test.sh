#!/bin/bash

dir="sampler"

multi_gpu() {
    python3 $dir/multigpu_minibatch.py --gpus $1 >logs/multigpu_minibatch$1gpu.log
    pid=$!
    wait $pid
    echo "$@"
    python3 $dir/multigpu_microbatch_base.py --gpus $1 >logs/multigpu_microbatch_base$1gpu.log
}

test() {
    multigpu 4
    # multigpu 2
    # multigpu 1
    # epochtimetest 4
    # epochtimetest 2
    # epochtimetest 1
}

single_gpu() {
    python $dir/gpu1_microbatch.py --num_epochs $1 >logs/gpu1_microbatch.log
    pid=$!
    wait $pid
    echo "$@"
    python $dir/gpu1_minibatch.py --num_epochs $1 >logs/gpu1_minibatch.log
}

epochtime_test() {
    python3 $dir/epochtimetest.py --gpus $1 >logs/epochtimetest$1gpu.log
    pid=$!
    wait $pid
    echo "$@"
}

clear() {
    rm logs/*gpu.log
}

# compare 4
multi_gpu 2
