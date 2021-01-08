import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Do not print info and warning messages

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ignore GPU devices

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--version",
                    type=str,
                    required=True,
                    default="big",  # togliere alla consegna
                    help="Version of the model (big or little)")

args = parser.parse_args()

assert args.version in ["big", "little"], "Error: parameter version is not correct"


def bigtraining():
    pass


def smalltraining():
    pass


if args.version == "big":
    bigTraining()
else:
    smallTraining()
