import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 					# ignore GPU devices

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot 				# for magnitude based pruning
import zlib 												# to compress the model
import tensorflow.lite as tflite
from SignalGenerator import SignalGenerator 

# NUMBER OF MODELS
N = 4


#### PARSING INPUT PARAMETERS ####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--version", type=int, required=True)
parser.add_argument("--qos", type=int, required=False, default=2)
args = parser.parse_args()
QOS = args.qos

if args.version == 0:
	MODEL = "CNN-0"
	EPOCHS = 20
	LEARNING_RATE = 0.01
	SEED = 42	
	
elif args.version == 1:
	MODEL = "CNN-1"
	EPOCHS = 20
	LEARNING_RATE = 0.01
	SEED = 42	
	
elif args.version == 2:
	MODEL = "DS-CNN-0"
	EPOCHS = 20
	LEARNING_RATE = 0.01
	SEED = 42	
	
elif args.version == 3:
	MODEL = "DS-CNN-1"
	EPOCHS = 20
	LEARNING_RATE = 0.01
	SEED = 42
	
else:
    raise ValueError(f"Version argument must be a number from 0 to {N-1}")
##################################################################################################

#### SEED DEFINITION #############################################################################
tf.random.set_seed(SEED)
np.random.seed(SEED)
##################################################################################################

#### FIXED PARAMETERS ############################################################################
MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}

SAMPLING_RATE = 16000


# MFCC PARAMS
options = MFCC_OPTIONS
strides = [2, 1]
input_shape=[49, 10, 1]

##################################################################################################


#### DATASET DOWNLOAD ############################################################################
zip_path = tf.keras.utils.get_file(
	origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
	fname="mini_speech_commands.zip",
	extract=True,
	cache_dir=".",
	cache_subdir="data")

data_dir = os.path.join(".", "data", "mini_speech_commands")
##################################################################################################


#### DATASET LOADING #############################################################################
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

# LABELS order is fixed, read from file
f = open("labels.txt", "r")
LABELS = f.read().split(" ")

generator = SignalGenerator(LABELS, SAMPLING_RATE, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)
##################################################################################################


#### MODELS ######################################################################################
if MODEL == "CNN-0":
	model = keras.Sequential([
		keras.layers.Conv2D(input_shape=input_shape, filters=int(128), kernel_size=[3, 3], strides=strides, use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=int(128), kernel_size=[3, 3], strides=[1, 1], use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=int(128), kernel_size=[3, 3], strides=[1, 1], use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.GlobalAveragePooling2D(),
		keras.layers.Dense(units=len(LABELS))
])

if MODEL == "CNN-1":
	model = keras.Sequential([
		keras.layers.Conv2D(input_shape=input_shape, filters=int(64), kernel_size=[3, 3], strides=strides, use_bias=True),
		keras.layers.BatchNormalization(momentum=0.2),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=int(64), kernel_size=[3, 3], strides=[1, 1], use_bias=True),
		keras.layers.BatchNormalization(momentum=0.2),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=int(64), kernel_size=[3, 3], strides=[1, 1], use_bias=True),
		keras.layers.BatchNormalization(momentum=0.2),
		keras.layers.ReLU(),
		keras.layers.GlobalAveragePooling2D(),
		keras.layers.Dense(units=len(LABELS))
])


if MODEL == "DS-CNN-0":
	model = keras.Sequential([
		keras.layers.Conv2D(input_shape=input_shape, filters=int(256), kernel_size=[3, 3], strides=strides, use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
		keras.layers.Conv2D(filters=int(256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
		keras.layers.Conv2D(filters=int(256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.GlobalAveragePooling2D(),
		keras.layers.Dense(units=len(LABELS))
])

if MODEL == "DS-CNN-1":
	model = keras.Sequential([
		keras.layers.Conv2D(input_shape=input_shape, filters=int(128), kernel_size=[3, 3], strides=strides, use_bias=True),
		keras.layers.BatchNormalization(momentum=0.2),
		keras.layers.ReLU(),
		keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=True),
		keras.layers.Conv2D(filters=int(128), kernel_size=[1, 1], strides=[1, 1], use_bias=True),
		keras.layers.BatchNormalization(momentum=0.2),
		keras.layers.ReLU(),
		keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=True),
		keras.layers.Conv2D(filters=int(128), kernel_size=[1, 1], strides=[1, 1], use_bias=True),
		keras.layers.BatchNormalization(momentum=0.2),
		keras.layers.ReLU(),
		keras.layers.GlobalAveragePooling2D(),
		keras.layers.Dense(units=len(LABELS))
])



##################################################################################################


#### BUILD MODEL PATH AND NAME ###################################################################
params_to_save = []

filename = f"{args.version}"

params_to_save.append(f"Model: {MODEL}\n")

params_to_save.append(f"epochs: {EPOCHS}\n")
params_to_save.append(f"lr: {LEARNING_RATE}\n")


filepath_base = f"./models_weights/version_{args.version}"

print(f"\nModel will be saved in {filepath_base}")
##################################################################################################


#### CALLBACKS ###################################################################################
checkpoint = tf.keras.callbacks.ModelCheckpoint(
	filepath = filepath_base+'/weights/'+filename,
	monitor='val_sparse_categorical_accuracy',
	patience=0,
	verbose=1,
	save_best_only=True,
	save_weights_only=True,
	mode='auto',
	save_freq='epoch',
)
callbacks = [checkpoint]
##################################################################################################


#### MODEL COMPILING #############################################################################
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
              )

model.summary()
params = model.count_params()
##################################################################################################


#### TRAIN AND EVALUATE MODEL ####################################################################
model.fit(train_ds, epochs=EPOCHS, verbose=1, validation_data=val_ds, callbacks=callbacks)

model.load_weights(filepath_base+'/weights/'+filename)
test_acc = model.evaluate(test_ds, verbose=1)[-1]
##################################################################################################


#### CONVERT TO TFLITE ############
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
##################################################################################################


#### SAVE MODEL ##################################################################################
if not os.path.exists("./models/"):
	os.makedirs("./models/")

with open("./models/"+filename+".tflite", 'wb') as fp:
	fp.write(tflite_model)
print(f"Final model size: {os.path.getsize('./models/'+filename+'.tflite')} bytes")
##################################################################################################


#### EVALUATE MODEL ON TEST SET ##################################################################
interpreter = tflite.Interpreter("models/"+filename+'.tflite')

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

accuracy = 0
count = 0
for x, y_true in test_ds.unbatch().batch(1):
	interpreter.set_tensor(input_details[0]['index'], x)
	interpreter.invoke()
	y_pred = interpreter.get_tensor(output_details[0]['index'])

	y_pred = y_pred.squeeze() #remove batch dim
	y_pred = np.argmax(y_pred)

	y_true = y_true.numpy().squeeze()

	accuracy += y_pred==y_true #1 if True, 0 otherwise
	count += 1

accuracy /= float(count)

print("Accuracy {}".format(accuracy))
##################################################################################################


#### SAVE ACCURACY ###############################################################################
f = open(filepath_base+f"/params_version_{args.version}.txt", "w")
f.write(f"{params}\n")
for params in params_to_save:
	f.write(params)

f.write(f"Final size: {os.path.getsize('models/'+filename+'.tflite')}\n")

f.write(f"Testset accuracy: {accuracy}")
f.close()
##################################################################################################
