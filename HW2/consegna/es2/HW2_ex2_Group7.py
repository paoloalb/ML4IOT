import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 					# ignore GPU devices

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot 				# for magnitude based pruning
import zlib 												# to compress the model
import tensorflow.lite as tflite

#### SEED DEFINITION #############################################################################
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
##################################################################################################

#### PARSING INPUT PARAMETERS ####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str)

args = parser.parse_args()

if args.version == "a":
	MODEL = "DS-CNN"
	MFCC = True
	EPOCHS = 20
	LEARNING_RATE = 0.0075
	QUANTIZATION = "W"
	STRUCTURED_W = 0.7
	MAGNITUDE_FS =  0.9

if args.version == "b":	
	MODEL = "DS-CNN"
	MFCC = True
	EPOCHS = 30
	LEARNING_RATE = 0.005
	QUANTIZATION = "W"
	STRUCTURED_W = 0.3
	MAGNITUDE_FS =  None
	
if args.version == "c":
	MODEL = "DS-CNN"
	MFCC = False
	EPOCHS = 30
	LEARNING_RATE = 0.01
	QUANTIZATION = "W"
	STRUCTURED_W = 0.4
	MAGNITUDE_FS =  None

##################################################################################################

#### FIXED PARAMETERS ############################################################################
STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}
MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}

SAMPLING_RATE = 16000


if MFCC is True:
    options = MFCC_OPTIONS
    strides = [2, 1]
    input_shape=[49, 10, 1]
else:
    options = STFT_OPTIONS
    strides = [2, 2]
    input_shape = [32, 32, 1]
##################################################################################################

#### SIGNAL GENERATOR CLASS ######################################################################
class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds
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

<<<<<<< Updated upstream
# hardcoded labels to avoid different orders based on different listdir implementation during testing
LABELS = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
print(LABELS)

=======
LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))
LABELS = LABELS[LABELS != "README.md"]
LABELS= ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
>>>>>>> Stashed changes
generator = SignalGenerator(LABELS, SAMPLING_RATE, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)
##################################################################################################


#### STRUCTURED PRUNING ##########################################################################
if STRUCTURED_W:
	ALPHA = STRUCTURED_W
else:
	ALPHA = 1
##################################################################################################


#### PT QUANTIZATION #############################################################################
def representative_dataset_gen():
	for x, _ in train_ds.take(1000):
		yield [x]
##################################################################################################


#### MODELS ######################################################################################
if MODEL == "MLP":
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=input_shape),
		keras.layers.Dense(units=int(256*ALPHA), activation="relu"),
		keras.layers.Dense(units=int(256*ALPHA), activation="relu"),
		keras.layers.Dense(units=int(256*ALPHA), activation="relu"),
		keras.layers.Dense(units=len(LABELS))
])
if MODEL == "CNN":
	model = keras.Sequential([
		keras.layers.Conv2D(input_shape=input_shape, filters=int(128*ALPHA), kernel_size=[3, 3], strides=strides, use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=int(128*ALPHA), kernel_size=[3, 3], strides=[1, 1], use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=int(128*ALPHA), kernel_size=[3, 3], strides=[1, 1], use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.GlobalAveragePooling2D(),
		keras.layers.Dense(units=len(LABELS))
])
if MODEL == "DS-CNN":		
	model = keras.Sequential([
		keras.layers.Conv2D(input_shape=input_shape, filters=int(256*ALPHA), kernel_size=[3, 3], strides=strides, use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
		keras.layers.Conv2D(filters=int(256*ALPHA), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
		keras.layers.Conv2D(filters=int(256*ALPHA), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
		keras.layers.BatchNormalization(momentum=0.1),
		keras.layers.ReLU(),
		keras.layers.GlobalAveragePooling2D(),
		keras.layers.Dense(units=len(LABELS))	
])
##################################################################################################


#### BUILD MODEL PATH AND NAME ###################################################################
params_to_save = []

filename = f"Group7_kws_{args.version}"

params_to_save.append(f"Model: {MODEL}\n")
params_to_save.append(f"mfcc: {MFCC}\n")
params_to_save.append(f"epochs: {EPOCHS}\n")
params_to_save.append(f"lr: {LEARNING_RATE}\n")

if QUANTIZATION:
	params_to_save.append(f"Quantization: {QUANTIZATION}\n")

if STRUCTURED_W:
	params_to_save.append(f"Structured_w: {STRUCTURED_W}\n")
	
if MAGNITUDE_FS:
	params_to_save.append(f"Magnitude_fs: {MAGNITUDE_FS}\n")


filepath_base = f"./kws_models/version_{args.version}"

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

# MAGNITUDE BASED PRUNING
if MAGNITUDE_FS:
	pruning_params = {'pruning_schedule':
						tfmot.sparsity.keras.PolynomialDecay(
							initial_sparsity=0.30,
							final_sparsity=MAGNITUDE_FS,
							begin_step=len(train_ds)*5,
							end_step=len(train_ds)*15
							)
	}
	
	callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
	
	prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
	model = prune_low_magnitude(model, **pruning_params)
	
	if MFCC:
		model.build([32, 49, 10])
	else: 
		model.build([32, 32, 32])
		
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
if MAGNITUDE_FS:
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
with open(filepath_base+"/"+filename+".tflite", 'wb') as fp:
	fp.write(tflite_model)
print(f"Final model size: {os.path.getsize(filepath_base+'/'+filename+'.tflite')} bytes")
	
 
with open(filepath_base+"/"+filename+".zlib", 'wb') as fp:
	tflite_compressed = zlib.compress(tflite_model)
	fp.write(tflite_compressed)
print(f"Compressed size: {os.path.getsize(filepath_base+'/'+filename+'.zlib')} bytes")
##################################################################################################


#### EVALUATE MODEL ON TEST SET ##################################################################
interpreter = tflite.Interpreter(filepath_base+'/'+filename+'.tflite')
    	
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

f.write(f"Final size: {os.path.getsize(filepath_base+'/'+filename+'.tflite')}\n")
f.write(f"Compressed final size: {os.path.getsize(filepath_base+'/'+filename+'.zlib')}\n")

f.write(f"Testset accuracy: {accuracy}")
f.close()
##################################################################################################
