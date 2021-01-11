import requests
import json
import base64
import numpy as np
from datetime import datetime
import tensorflow as tf

from scipy.stats import entropy

def SuccessChecker_BinEntropy(inf_array, threshold):
	if entropy(inf_array, base=2) <= threshold:
		return True
	else:
		return False

#####################################################

COMM_COST = 0
URL = "http://192.168.1.170:8080"

def BigRequest(url, file_path):
	
	dateTimeObj = datetime.now()
	timestamp = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")

	wav = tf.io.read_file(file_path).numpy()
	encoded_audio = base64.b64encode(wav).decode()

	json_audio = {"n": "audio", "u": "pi", "t": timestamp, "v": encoded_audio}
	out = {"bn": "little_service", "e": json_audio}  
	
	try:
    		r = requests.post(url = url, json = out)
	except requests.exceptions.RequestException as err:  
    		raise SystemExit(err)
	
	body = r.json()
	sample_label = body["e"]["label"]
	return (sample_label, len(json.dumps(out)))
	
	
##################################################### Main di prova

for i in range(30):
	path = "raw_data/yes1.wav"
	inference = np.array([0.5,0.5])
	if not SuccessChecker_BinEntropy(inference, 0.8):	
		label, cost = BigRequest(URL, path)
		print(label)
		if label != "ERROR":
			COMM_COST += cost

print("Communication cost: {}MB".format(np.round(COMM_COST/1000000),1))

