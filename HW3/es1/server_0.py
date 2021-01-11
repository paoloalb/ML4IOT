import cherrypy
import json

import sys
import os
import io
import numpy as np
import base64


class BigService(object): 
	exposed= True 
	
	def POST(self, *path, **query): 
		
		json_obj = cherrypy.request.body.read()
		dict_obj = json.loads(json_obj)
		audio = np.frombuffer(base64.b64decode(dict_obj["e"]["v"]), dtype=np.int16)

		interpreter = tflite.Interpreter('./models/Group7_big.tflite')

		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		interpreter.set_tensor(input_details[0]['index'], audio)
		interpreter.invoke()
		y_pred = interpreter.get_tensor(output_details[0]['index'])

		y_pred = y_pred.squeeze()  # remove batch dim

		f = open("labels.txt", "r")
		LABELS = f.read().split(" ")
		f.close()

		sample_label = {LABELS[i]: y_pred[i] for i, j in enumerate(LABELS)}

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