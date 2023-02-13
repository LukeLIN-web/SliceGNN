
from enum import Enum


class RunMode(Enum):
    NORMAL = 0


def gpu(device_id=0):
    return 'cuda:{:}'.format(device_id)
