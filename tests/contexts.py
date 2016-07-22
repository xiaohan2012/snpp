import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import snpp

def abs_path(rel_path):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, rel_path)
