
import time
from enum import Enum


class RunMode(Enum):
    NORMAL = 0


def gpu(device_id=0):
    return 'cuda:{:}'.format(device_id)


def get_default_common_config(run_mode: RunMode = RunMode.NORMAL):
    default_common_config = {}
    default_common_config['_run_mode'] = run_mode
    default_common_config['num_sample_worker'] = 1
    default_common_config['num_train_worker'] = 1
    default_common_config['num_epoch'] = 6
    default_common_config['batch_size'] = 1024
    default_common_config['num_hidden'] = 256

    return default_common_config


def add_common_arguments(argparser, run_config):
    run_mode = run_config['_run_mode']


def print_run_config(run_config):
    print('config:eval_tsp="{:}"'.format(time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())))
    for k, v in run_config.items():
        if not k.startswith('_'):
            print('config:{:}={:}'.format(k, v))

    for k, v in run_config.items():
        if k.startswith('_'):
            print('config:{:}={:}'.format(k, v))


def process_common_config(run_config):
    run_config['train_workers'] = [
        gpu(i) for i in range(run_config['num_train_worker'])]
    run_config['sample_workers'] = [gpu(
        run_config['num_train_worker'] + i) for i in range(run_config['num_sample_worker'])]
