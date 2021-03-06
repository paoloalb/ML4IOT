import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ignore GPU

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot  # for magnitude based pruning
from time import gmtime, strftime  # to put date on the file name
import zlib  # to compress the model

#### SEED DEFINITION AND PARAMETERS #############################################################################
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

EPOCHS = 20
##################################################################################################

#### PARSING INPUT PARAMETERS ####################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, required=True)

args = parser.parse_args()

if str(args.version).lower() == "a":
    MODEL = "CNN"
    EPOCHS = 20
    LEARNING_RATE = 0.001
    QUANTIZATION = "W"
    ALPHA = 0.12
    MAGNITUDE_FS = 0.745
    COMPRESSED = True

elif str(args.version).lower() == "b":
    MODEL = "CNN"
    EPOCHS = 20
    LEARNING_RATE = 0.001
    QUANTIZATION = "W"
    ALPHA = 0.07
    MAGNITUDE_FS = 0.75
    COMPRESSED = True
else:
    raise ValueError("Version argument must be a or b")

##################################################################################################


#### MAE METRIC CLASS ######################################################################
class MAEmetric(tf.keras.metrics.Metric):
    temp_metric = None

    # is possible to handle one of the two metrics, by changing the parameter metric
    def __init__(self, name, metric_type, **kwargs):
        super(MAEmetric, self).__init__(name=name, **kwargs)
        self.total = self.add_weight("total", initializer="zeros", shape=[2])
        self.count = self.add_weight("total", initializer="zeros")
        self.metric_type = metric_type

    def update_state(self, y_true, y_pred, sample_weight=None):
        # update the internal MeanAbsoluteError instance
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=[0, 1])
        self.total.assign_add(error)
        self.count.assign_add(1)
        return

    def result(self):
        assert (self.metric_type.upper() in ["T", "H", "T+H"]), "Error. mertic type must be T or H"
        switcher = {
            "T": 0,
            "H": 1,
            "T+H": 2}
        idx = switcher.get(self.metric_type.upper(), "Invalid argument")

        if idx == 2:  # If i chose T+H as the metric_type argument, i calculate the mean of the 2 metrics
            return (tf.math.divide_no_nan(self.total, self.count)[0] + tf.math.divide_no_nan(self.total, self.count)[
                1]) / 2
        else:
            return tf.math.divide_no_nan(self.total, self.count)[idx]

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return


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

        return ds


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
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)
##################################################################################################



#### PT QUANTIZATION #############################################################################
def representative_dataset_gen():
    for x, _ in train_ds.take(1000):
        yield [x]


##################################################################################################

if MODEL == "MLP":
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(6, 2)),
        keras.layers.Dense(int(128 * ALPHA), activation='relu', name='first_dense'),
        keras.layers.Dense(int(128 * ALPHA), activation='relu', name='second_dense'),
        keras.layers.Dense(12, name='output'),
        keras.layers.Reshape((6, 2), input_shape=(12,))
    ])

elif MODEL == "CNN":
    model = keras.Sequential([
        keras.layers.Conv1D(filters=int(64 * ALPHA), kernel_size=3, activation="relu", name="conv1",
                            input_shape=(6, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=int(64 * ALPHA), activation="relu", name="first_dense"),
        keras.layers.Dense(units=12, name='output'),
        keras.layers.Reshape((6, 2), input_shape=(12,))

    ])
else:
    raise ValueError("Model not found: " + str(MODEL))

filename = f"Group7_th_{args.version}"

filepath_base = f"./th_models/version_{args.version}"

print(f"\nModel will be saved in {filepath_base}")


#### CALLBACKS ###################################################################################
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath_base + '/weights/' + filename,
    monitor='val_temp_hum_MAE',
    patience=0,
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch',
)

callbacks = [checkpoint]

# MAGNITUDE BASED PRUNING
pruning_params = {'pruning_schedule':
    tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.30,
        final_sparsity=MAGNITUDE_FS,
        begin_step=len(train_ds) * 5,
        end_step=len(train_ds) * 15
    )
}

callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model = prune_low_magnitude(model, **pruning_params)

model.build([32, 6, 2])

##################################################################################################

#### MODEL COMPILING #############################################################################
temp_metric = MAEmetric(name="TempMAE", metric_type="T")
hum_metric = MAEmetric(name="HumMAE", metric_type="H")
# We also add a third metric, that is the average of the two (for checkpointing)
temp_hum_metric = MAEmetric(name="temp_hum_MAE", metric_type="T+H")

metric = [temp_hum_metric, temp_metric, hum_metric]

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.MSE,
              metrics=metric)

model.summary()
params = model.count_params()
##################################################################################################

#### TRAIN AND EVALUATE MODEL ####################################################################
model.fit(train_ds, epochs=EPOCHS, verbose=1, validation_data=val_ds, callbacks=callbacks)

# model.load_weights("models/model/best_0.565566")
model.load_weights(filepath_base + '/weights/' + filename)

test_MAE = model.evaluate(test_ds, verbose=1)[-2:]  # 2->labels
##################################################################################################

#### CONVERT TO TFLITE ############
model = tfmot.sparsity.keras.strip_pruning(model)

# POST TRAINING QUANTIZATION
converter = tf.lite.TFLiteConverter.from_keras_model(model)
if QUANTIZATION == "W":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

else:
    if QUANTIZATION == "W+A":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
##################################################################################################

#### SAVE MODEL ##################################################################################
with open(filepath_base + "/" + filename + ".tflite", 'wb') as fp:
    fp.write(tflite_model)
print(f"Final model size: {os.path.getsize(filepath_base + '/' + filename + '.tflite')} bytes")

if COMPRESSED:
    with open(filepath_base + "/" + filename + ".zlib", 'wb') as fp:
        tflite_compressed = zlib.compress(tflite_model)
        fp.write(tflite_compressed)
    print(f"Compressed size: {os.path.getsize(filepath_base + '/' + filename + '.zlib')} bytes")
##################################################################################################

#### SAVE ACCURACY ###############################################################################
f = open(filepath_base + "/RESULT.txt", "w")
f.write(f"{params}\n{test_MAE}")
f.close()
##################################################################################################

