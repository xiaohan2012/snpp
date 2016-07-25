import os
import sys
import random
import numpy as np


sys.path.insert(0, os.path.abspath('..'))

import snpp


def abs_path(rel_path):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, rel_path)


def reset_random_seed():
    random.seed(12345)
    np.random.seed(12345)
