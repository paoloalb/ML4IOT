import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Do not print info and warning messages
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ignore GPU devices

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--version",
                    type=str,
                    #required=True,
                    default="big",  # togliere il default alla consegna
                    help="Version of the model (big or little)")

args = parser.parse_args()

assert args.version in ["big", "little"], "Error: parameter version is not correct"


def bigtraining():
    
    pass # mfcc


def littletraining():
    pass #stft


if args.version == "big":
    bigtraining()
else:
    littletraining()

zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname="mini_speech_commands.zip",
    extract=True,
    cache_dir=".",
    cache_subdir="data")

data_dir = os.path.join(".", "data", "mini_speech_commands")

train_files, val_files, test_files = [], [], []

with open("kws_train_split.txt", "r") as train_file:
    for filename in train_file:
        train_files.append(filename[:-1])

with open("kws_val_split.txt", "r") as val_file:
    for filename in val_file:
        val_files.append(filename[:-1])

with open("kws_test_split.txt", "r") as test_file:
    for filename in test_file:
        test_files.append(filename[:-1])

print(f"Train set size: {len(train_files)}")
print(f"Val set size: {len(val_files)}")
print(f"Test set size: {len(test_files)}")

f = open("labels.txt", "r")
LABELS = f.read().split(" ")
f.close()

