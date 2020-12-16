import argparse
import numpy as np
import os
from WindowGenerator import WindowGenerator
from PairMAE import PairMAE

import pandas as pd
import tensorflow as tf
from tensorflow import keras

# GPU acceleration setup
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# input params definition
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model name', default="CNN")
args = parser.parse_args()


EPOCHS = 1
seed = 42

# setting seed
tf.random.set_seed(seed)
np.random.seed(seed)

# download dataset 
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)

# load dataset into datafme
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

# split into train, val and test
n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

# from sklearn.model_selection import train_test_split
# train_data, test_val_data = train_test_split(data, test_size=0.3, random_state=1)
# test_data, val_data = train_test_split(test_val_data, test_size=1/3, random_state=1)

# eval mean and standard dev
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

# window width
input_width = 6

# generate windows in batches inside tf.Dataset
generator = WindowGenerator(input_width, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)



# build model
if args.model == "MLP":
	model = keras.Sequential([
    keras.layers.Flatten(input_shape=(6, 2)),
    keras.layers.Dense(128, activation='relu', name='first_dense'),
    keras.layers.Dense(128, activation='relu', name='second_dense'),
    keras.layers.Dense(12, name='output'),
    keras.layers.Reshape((6, 2), input_shape=(12,))
])

if args.model == "CNN":
	model = keras.Sequential([
	keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", name="conv1", input_shape=(6, 2)),
	keras.layers.Flatten(),
	keras.layers.Dense(units=64, activation="relu", name="first_dense"),
	keras.layers.Dense(units=12, name='output'),
    keras.layers.Reshape((6, 2), input_shape=(12,))

    ])

mae_temp = PairMAE(name="temp_MAE", metric=0)
mae_hum = PairMAE(name="hum_MAE", metric=1)
metrics = [mae_temp, mae_hum]

model.compile(optimizer='adam',
              loss=tf.keras.losses.MSE,
              metrics=metrics)


model.summary()
params = model.count_params()


# train and evaluate model
model.fit(train_ds, epochs=EPOCHS, verbose=1, validation_data=val_ds)
test_MAE = model.evaluate(test_ds, verbose=1)[-args.labels:]
