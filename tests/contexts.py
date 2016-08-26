# referred from here:
# https://github.com/kawadia/pyspark.test/blob/master/examples/conftest.pyhttps://github.com/kawadia/pyspark.test/blob/master/examples/conftest.py

import findspark  # this needs to be the first import
findspark.init()

import os
import sys
import random
import numpy as np
import logging
import pytest

from pyspark import SparkConf
from pyspark import SparkContext

sys.path.insert(0, os.path.abspath('..'))

import snpp


def abs_path(rel_path):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, rel_path)


def reset_random_seed():
    random.seed(12345)
    np.random.seed(12345)


def quiet_py4j():
    """ turn down spark logging for the test context """
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def spark_context(request):
    """ fixture for creating a spark context
    Args:
    request: pytest.FixtureRequest object
    """
    conf = (SparkConf().setMaster("local[2]").setAppName("SparkTest"))
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir('checkpoint')  # Stackoverflow error
    request.addfinalizer(lambda: sc.stop())

    quiet_py4j()
    return sc
