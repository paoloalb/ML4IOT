from MyMQTT import MyMQTT
from MQTT_Handler import MQTT_Handler
from SignalGenerator import SignalGenerator
import json
import base64
import time
import requests
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 					# ignore GPU devices
import zipfile
import numpy as np

import tensorflow as tf


######### DEFINE MQTT CONNECTION PARAMETERS #################################################################################
topic_root = "276545"
recording_topic = topic_root+"/recording"	# one topic to send recording data and one topic to read the client's predictions
preds_topic = topic_root+"/predictions"

clientID = "cooperative_client"
#############################################################################################################################


N = 4  # number of clients
predictions = []


######### DEFINE SUBCRIPTION HANDLER ########################################################################################
class myHandler(MQTT_Handler):
	def notify(self, topic, msg):
		if topic == preds_topic:
			data = json.loads(msg)
			print(f"Received inference from {data['bn']}, recording id: {data['record_id']}")
			logits = np.frombuffer(base64.b64decode(bytes(data["e"][0]["vd"], "utf-8")), dtype=np.float32)
			
			predictions.append(list(logits))
#############################################################################################################################

'''			
def download_dataset(url):
	print("Downloading the dataset")
	r = requests.get(url)
	open("mini_speech_commands.zip", 'wb').write(r.content)
	
	print("Unzipping the dataset")
	with zipfile.ZipFile("mini_speech_commands.zip", "r") as zip_ref:
		zip_ref.extractall("data")
		
	print("Removing zip file")
	os.remove("mini_speech_commands.zip")
'''			
	
######## START MQTT CLIENT ##################################################################################################		
handler = myHandler(clientID)						# init the handler
handler.run()										# start the MQTT client
handler.myMqttClient.mySubscribe(preds_topic)		# subscribe to predictions topic
#############################################################################################################################


######## DOWNLOAD DATASET ###################################################################################################
'''
if not os.path.exists("./data"):
	download_dataset("http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip")
'''
zip_path = tf.keras.utils.get_file(
	origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
	fname="mini_speech_commands.zip",
	extract=True,
	cache_dir=".",
	cache_subdir="data")

data_dir = os.path.join(".", "data", "mini_speech_commands")
#############################################################################################################################


######## LOAD DATASET #######################################################################################################
test_files = []
# read test set
with open("kws_test_split.txt", "r") as test_file:
	for filename in test_file:
		test_files.append(filename[:-1])

# LABELS order is fixed, read from file
f = open("labels.txt", "r")
LABELS = f.read().split(" ")


# set preprocessing options
MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}

# read and preprocess test data
generator = SignalGenerator(LABELS, sampling_rate=16000, **MFCC_OPTIONS)
test_ds = generator.make_dataset(test_files, False)
############################################################################################################################


######## COOPERATIVE INFERENCE #############################################################################################
correct = 0
count = 0

# cycle through all the test set
for x, y_true in test_ds.unbatch().batch(1):
	print(f"{count+1}/{len(test_files)}")
	encoded_audio = base64.b64encode(x).decode()
		
	audio = {"n": "audio", "t":int(time.time()), "vd": encoded_audio}
	message = {"bn": "cooperative_client", "record_id":count, "e": [audio]}
	handler.myMqttClient.myPublish(recording_topic, json.dumps(message))
		
	while len(predictions)!=N:
		time.sleep(0.001)
	
	predictions = np.asarray(predictions)
	predictions = predictions.sum(axis=0)
	pred = np.argmax(predictions)
	
	if pred == y_true.numpy().squeeze():
		correct += 1
		
	count += 1

	print(f"Accuracy: {correct/count:.4f}")
	predictions = []
#############################################################################################################################
print(f"Accuracy: {correct/count:.2f}")

handler.end()
			
		
