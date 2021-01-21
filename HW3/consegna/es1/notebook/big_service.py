import cherrypy
import json
from scipy import signal
import sys
import os
import io
import numpy as np
import base64
import tensorflow.lite as tflite
import tensorflow as tf
from datetime import datetime


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
        tic = time.time()
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


os.environ["CUDA_VISIBLE_DEVICES"] = "" 					# ignore GPU devices

SAMPLING_RATE = 16000


MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}
        
f = open("labels.txt", "r")
LABELS = f.read().split(" ")
        
generator = SignalGenerator(LABELS, SAMPLING_RATE, **MFCC_OPTIONS)

class BigService(object): 
	exposed = True
	
	def __init__(self):
		self.interpreter = tflite.Interpreter('Group7_big.tflite')

		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()

	def POST(self, *path, **query): 
		
		json_obj = cherrypy.request.body.read()
		dict_obj = json.loads(json_obj)
		audio = np.frombuffer(base64.b64decode(dict_obj["e"]["vd"]), dtype=np.int16)
		#audio = np.frombuffer(base64.b85decode(dict_obj["e"]["v"]), dtype=np.int16)
		
		# preprocessing
		audio = bytes(audio)
		audio, _ = tf.audio.decode_wav(audio)
		audio = tf.squeeze(audio, axis=1)

		audio = generator.pad(audio)
		spectrogram = generator.get_spectrogram(audio)
		mfccs = generator.get_mfccs(spectrogram)
		data = tf.expand_dims(mfccs, -1)
		data = tf.expand_dims(data, 0)
		self.interpreter.set_tensor(self.input_details[0]['index'], data)
		self.interpreter.invoke()
		y_pred = self.interpreter.get_tensor(self.output_details[0]['index'])

		y_pred = y_pred.squeeze()  # remove batch dim

		sample_label = {"label":str(np.argmax(np.array(y_pred)))}
		print(sample_label)	
		
		return json.dumps(sample_label)


if __name__ =='__main__': 
	conf={ '/': { 'request.dispatch': cherrypy.dispatch.MethodDispatcher(), 
			'tools.sessions.on': True, } 
		} 

	cherrypy.tree.mount(BigService(), '/', conf)
	cherrypy.config.update({'server.socket_host': '0.0.0.0'})
	cherrypy.config.update({'server.socket_port': 8080})
	cherrypy.engine.start()
	cherrypy.engine.block()
