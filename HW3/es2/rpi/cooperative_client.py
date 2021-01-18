from MyMQTT import MyMQTT
from MQTT_Handler import MQTT_Handler
from SignalGenerator import SignalGenerator

import json
import base64
import time
import requests
import os
import numpy as np
import tensorflow as tf
import argparse
import queue
import threading

######### DEFINE MQTT CONNECTION PARAMETERS #################################################################################
topic_root = "276545"
recording_topic = topic_root+"/recording"	# one topic to send recording data and one topic to read the client's predictions
preds_topic = topic_root+"/predictions"

clientID = "cooperative_client"
#############################################################################################################################


N = 4  # number of clients
inference_queue = list()


######### DEFINE SUBCRIPTION HANDLER ########################################################################################
class myHandler(MQTT_Handler):
	def notify(self, topic, msg):
		if topic == preds_topic:
			data = json.loads(msg)
			#print(f"Received inference from {data['device_id']}, recording id: {data['record_id']}")	# logging
			
			# decode the message (base64 string -> numpy array)
			logits = np.frombuffer(base64.b64decode(data["logits"]), dtype=np.float32)
			
			inference_queue.append((data['record_id'], list(logits)))
#############################################################################################################################


QOS = 0	# quality of service is set to 0
	
	
######## START MQTT CLIENT ##################################################################################################		
handler = myHandler(clientID)						# init the handler
handler.run()										# start the MQTT client
handler.myMqttClient.mySubscribe(preds_topic, QOS)	# subscribe to predictions topic
#############################################################################################################################


######## DOWNLOAD DATASET ###################################################################################################
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


######### THREAD WORKER ####################################################################################################
def send_recordings(ds, handler, topic, QOS):
	count=0
	# cycle through all the test set
	for x in ds:
	
		encoded_audio = base64.b64encode(x).decode()			# encode sample
		
		# create the message in SenML
		audio = {"n": "audio", "t": 0, "u":"/", "vd": encoded_audio}
		message = {"bn": "cooperative_client", "bt": int(time.time()), "record_id":count, "e": [audio]}
		
		handler.myMqttClient.myPublish(topic, json.dumps(message), QOS)		# publish the message on the recording topic
		#print(f"Recording #{count} published under topic {topic}")
		
		count += 1
		time.sleep(0.01)
	return
############################################################################################################################


######## COOPERATIVE INFERENCE #############################################################################################

print("Loading dataset")
# storing the dataset
true_labels = []
ds = []
for i, (x, y_true) in enumerate(test_ds.unbatch().batch(1)):
	true_labels.append(y_true.numpy().squeeze())
	ds.append(x)
	print(f"{i}/{len(test_files)}", end="\r")

del test_ds

inference_queue = []	# empty queue 

# start the publisher thread
print("Starting publisher thread")
publisher = threading.Thread(target=send_recordings, args=(ds, handler, recording_topic, QOS))
publisher.start()

# evaluation of the accuracy
correct = 0
current_id = 0	

timeout_count = 0	# timeout count used to resend recording after timeout

# take the predictions and compute the mean of the logits (the sum in enough because the argmax returns the same result)

# while we have evaluated all the samples
while current_id<len(test_files):
	replies = []
	predictions = []
	
	# retrieve the inferences related to the current recording 
	for ix, pred in enumerate(inference_queue):
		if pred[0]==current_id:
			replies.append(ix)
	

	if len(replies) == N:	# all the devices have aswered with their inference
		timeout_count = 0
		for reply in replies[::-1]:
			predictions.append(inference_queue.pop(reply)[1])
				
		predictions = np.asarray(predictions)
		predictions = predictions.sum(axis=0)
		pred = np.argmax(predictions)			# take the final cooperative prediction
			
		# count how many correct
		if pred == true_labels[current_id]:
			correct += 1
				
		current_id += 1 	# go to the next recording to evaluate

		print(f"{current_id-1}/{len(test_files)} Current accuracy: {correct/current_id:.4f}")		# logging
	else:
		timeout_count += 1
		
		# if the timeout is reached, remove the related inferences (if any) from the queue and resend
		if timeout_count >= 2//0.001:
			timeout_count = 0
			print(f"TIMEOUT reached, resending record #{current_id}, only {len(replies)} inferences")
			# remove partial inference messages
			for reply in replies[::-1]:
				inference_queue.pop(reply)
				
			# resend the message
			encoded_audio = base64.b64encode(ds[current_id]).decode()			# encode sample
			
			# create the message in SenML
			audio = {"n": "audio", "t": 0, "u":"/", "vd": encoded_audio}
			message = {"bn": "cooperative_client", "bt": int(time.time()), "record_id":current_id, "e": [audio]}
			
			handler.myMqttClient.myPublish(recording_topic, json.dumps(message), QOS)		# publish the message on the recording topic
	
	time.sleep(0.001)
#############################################################################################################################


print(f"Accuracy: {correct/current_id:.4f}")

publisher.join()	# wait until the pulisher ends (this happens for sure before)
handler.end()		# terminate MQTT client
		
