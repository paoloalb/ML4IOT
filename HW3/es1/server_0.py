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


RESAMPLING_RATE = 16000


linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins = 40,
        num_spectrogram_bins = 321,
        sample_rate = RESAMPLING_RATE,
        lower_edge_hertz = 20,
        upper_edge_hertz = 4000)


def resample(recording, new_sampling_rate):
	resampled_audio = signal.resample_poly(recording, 1, 3)
	resampled_audio = resampled_audio.astype(np.int16)
	return resampled_audio


def pad(audio):
	zero_padding = tf.zeros([RESAMPLING_RATE] - tf.shape(audio), dtype=tf.float32)
	audio = tf.concat([audio, zero_padding], 0)
	audio.set_shape([RESAMPLING_RATE])

	return audio


def get_spectrogram(resampled, frame_length, frame_step):
	# convert to tensor
	tf_audio = tf.constant(resampled, dtype=np.float32)

	# scale to [-1 1]
	normalized_tf_audio = tf.math.divide(
		tf.add(
			tf_audio,
			32768
		),
		tf.constant(65535, dtype=float),
	)

	normalized_tf_audio = tf.math.subtract(normalized_tf_audio, 0.5)
	normalized_tf_audio = tf.multiply(normalized_tf_audio, 2)

	# compute stft and spectrogram
	stft = tf.signal.stft(normalized_tf_audio, frame_length=frame_length, frame_step=frame_step,
						  fft_length=frame_length)
	spectrogram = tf.abs(stft)

	return spectrogram

def get_mfcc(spectrogram):

    mel_spectrogram = tf.tensordot(
        spectrogram,
        linear_to_mel_weight_matrix,
        1)
    mel_spectrogram.set_shape((49, 40)) # spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # compute mfcc fom mel spectrogram and take first 10 coefficients
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :10]
    return mfccs

def preprocess_with_mfcc(audio, frame_length, frame_step):
	spectrogram = get_spectrogram(audio, frame_length, frame_step)

	mfccs = get_mfcc(spectrogram)
	mfccs = tf.expand_dims(mfccs, -1)
	mfccs = tf.expand_dims(mfccs, 0)
	return mfccs

class BigService(object): 
	exposed= True
	
	f = open("labels.txt", "r")
	LABELS = f.read().split(" ")
	f.close()

	def POST(self, *path, **query): 
		
		json_obj = cherrypy.request.body.read()
		dict_obj = json.loads(json_obj)
		audio = np.frombuffer(base64.b64decode(dict_obj["e"]["v"]), dtype=np.float32)
		#resampled = resample(audio, RESAMPLING_RATE)
		#resampled = tf.cast(audio, dtype=tf.float32)
		resampled = pad(audio)
		frame_length = int(0.040 * RESAMPLING_RATE)
		frame_step = int(0.020 * RESAMPLING_RATE)
		data = preprocess_with_mfcc(resampled, frame_length, frame_step)


		interpreter = tflite.Interpreter('./models/Group7_big.tflite')

		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		interpreter.set_tensor(input_details[0]['index'], data)
		interpreter.invoke()
		y_pred = interpreter.get_tensor(output_details[0]['index'])

		y_pred = y_pred.squeeze()  # remove batch dim

		print(y_pred)

		sample_label = {"label":str(np.argmax(np.array(y_pred)))}

		out = {"bn": "big_service", "e": sample_label}
		
		return json.dumps(out)


if __name__ =='__main__': 
	conf={ '/': { 'request.dispatch': cherrypy.dispatch.MethodDispatcher(), 
			'tools.sessions.on': True, } 
		} 

	cherrypy.tree.mount(BigService(), '/', conf)
	cherrypy.config.update({'server.socket_host': '0.0.0.0'})
	cherrypy.config.update({'server.socket_port': 8080})
	cherrypy.engine.start()
	cherrypy.engine.block()