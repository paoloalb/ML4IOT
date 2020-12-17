import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 					# ignore GPU

import tensorflow as tf
import tensorflow.lite as tflite
import numpy as np
import zlib 												# to compress the model
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--mfcc", action="store_true")
parser.add_argument("--compressed", action="store_true")

args = parser.parse_args()

#### FIXED PARAMETERS ############################################################################

##################################################################################################

#### WINDOW GENERATOR CLASS ######################################################################
class WindowGenerator:
    def __init__(self, input_width, mean, std):
        self.input_width = input_width
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])  # one for tem and one for hum
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])  # other dimension to match data shape

    def split_window(self, features):
        inputs = features[:, :-6, :]

        labels = features[:, -6:, :]
        num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, 6, num_labels])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.input_width + 6,
            sequence_stride=1,
            batch_size=32)


        ds = ds.map(self.preprocess)

        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)


        return ds.prefetch(tf.data.experimental.AUTOTUNE)
##################################################################################################


#### DATASET DOWNLOAD ############################################################################
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.',
    cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
##################################################################################################


#### DATASET LOADING #############################################################################
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

# split into train, val and test
n = len(data)
train_data = data[0:int(n * 0.7)]
val_data = data[int(n * 0.7):int(n * 0.9)]
test_data = data[int(n * 0.9):]

# eval mean and standard dev
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

# window width
input_width = 6

# generate windows in batches inside tf.Dataset
generator = WindowGenerator(input_width, mean, std)
test_ds = generator.make_dataset(test_data, False)
##################################################################################################

if args.compressed:
	with open(args.model, 'rb') as fp:
		model_zip = zlib.decompress(fp.read())
		interpreter = tflite.Interpreter(model_content=model_zip)
else:
	interpreter = tflite.Interpreter(args.model)
    	
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

mae_temp, mae_hum = 0, 0
count = 0
for x, y_true in test_ds.unbatch().batch(1):
	interpreter.set_tensor(input_details[0]['index'], x)
	interpreter.invoke()
	y_pred = interpreter.get_tensor(output_details[0]['index'])
		
	y_pred = y_pred.squeeze() #remove batch dim
	
	mae = abs(y_pred-y_true).numpy().mean(axis=1)
	mae_temp += mae[0][0]
	mae_hum += mae[0][1]	
	count += 1

mae_temp /= count
mae_hum /= count

print(f"MAE temp {mae_temp}, MAE hum {mae_hum}")
