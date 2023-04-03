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

Loader_reddit_pyg(){
    python3 $dir/loader_reddit_pyg.py --gpus $1  > logs/loader_reddit_pyg_gpu$1.log
    pid=$!
    wait $pid
}

run(){
    pkill python
    sleep 1
    python $1 nano_pergpu=$2
    pid=$!
    wait $pid
    echo "$@"
}

gasbench(){
    run gas_microbatchbenchmark.py 2
    run gas_microbatchbenchmark.py 4
    run gas_microbatchbenchmark.py 6
    run gas_microbatchbenchmark.py 8
}

microbench(){
    run microbatchbenchmark.py 2
    run microbatchbenchmark.py 4
    run microbatchbenchmark.py 6
    run microbatchbenchmark.py 8
}

quiverbench(){
    run quiver_microbatchbenchmark.py 2
    run quiver_microbatchbenchmark.py 4
    run quiver_microbatchbenchmark.py 6
    run quiver_microbatchbenchmark.py 8
}

# gasbench
microbench
