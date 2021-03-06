import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 					# ignore GPU

import tensorflow as tf
import tensorflow.lite as tflite
import numpy as np
import zlib 
import sys												# to compress the model

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str)

args = parser.parse_args()

#### FIXED PARAMETERS ############################################################################
MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}

SAMPLING_RATE = 16000

N = 4

options = MFCC_OPTIONS
strides = [2, 1]
input_shape=[49, 10, 1]
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
test_files = []

with open("kws_test_split.txt", "r") as test_file:
	for filename in test_file:
		test_files.append(filename[:-1])

print(f"Test set size: {len(test_files)}")

# LABELS order is fixed, read from file
f = open("labels.txt", "r")
LABELS = f.read().split(" ")

generator = SignalGenerator(LABELS, SAMPLING_RATE, **options)
test_ds = generator.make_dataset(test_files, False)
##################################################################################################



accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


preds = []
y = []
count = 0
for version in range(N):
	
	version_preds = []
	
	model_path = args.models+f"/{version}.tflite"
	
	interpreter = tflite.Interpreter(model_path)
	print(f"Loaded model version {version}")
    
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	
	for x, y_true in test_ds.unbatch().batch(1):
		interpreter.set_tensor(input_details[0]['index'], x)
		interpreter.invoke()
		y_pred = interpreter.get_tensor(output_details[0]['index'])
			
		y_pred = y_pred.squeeze() #remove batch dim

		version_preds.append(y_pred)	
		
		if version==0:
			y.append(y_true)
			count += 1
		
	preds.append(version_preds)

preds = np.asarray(preds)
preds = preds.transpose(1, 0, 2)
preds = preds.sum(axis=1)

preds = np.argmax(preds, axis=1)

y=np.asarray(y)

print(f"Accuracy: {(preds==y.T).sum()/count}")

