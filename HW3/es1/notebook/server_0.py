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
from SignalGenerator import SignalGenerator
os.environ["CUDA_VISIBLE_DEVICES"] = "" 					# ignore GPU devices

SAMPLING_RATE = 16000


MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}
        
f = open("labels.txt", "r")
LABELS = f.read().split(" ")
        
generator = SignalGenerator(LABELS, SAMPLING_RATE, **MFCC_OPTIONS)

def int16_to_float32(array):
	array = array/32768
	array = np.asarray(array, dtype=np.float32)
	return array


class BigService(object): 
	exposed= True
	
	f = open("labels.txt", "r")
	LABELS = f.read().split(" ")
	f.close()

	def POST(self, *path, **query): 
		
		json_obj = cherrypy.request.body.read()
		dict_obj = json.loads(json_obj)
		audio = np.frombuffer(base64.b64decode(dict_obj["e"]["v"]), dtype=np.int16)
		
		audio = int16_to_float32(audio)
		
		audio = generator.pad(audio)
		spectrogram = generator.get_spectrogram(audio)
		mfccs = generator.get_mfccs(spectrogram)
		data = tf.expand_dims(mfccs, -1)

		interpreter = tflite.Interpreter('./models/Group7_big.tflite')

		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		interpreter.set_tensor(input_details[0]['index'], [data])
		interpreter.invoke()
		y_pred = interpreter.get_tensor(output_details[0]['index'])

		y_pred = y_pred.squeeze()  # remove batch dim

		sample_label = {"label":str(np.argmax(np.array(y_pred)))}
		
		print(sample_label)	
		
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
