from MyMQTT import MyMQTT
from MQTT_Handler import MQTT_Handler
import tensorflow as tf
import base64
import json
import argparse
import numpy as np
import time


#### PARSING INPUT PARAMETERS ####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--qos", type=int, required=False, default=2)

args = parser.parse_args()
QOS = args.qos
##################################################################################################


#### LOAD MODEL ##################################################################################
interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors() #allocate ram

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
##################################################################################################


######### DEFINE MQTT CONNECTION PARAMETERS ######################################################
topic_root = "276545"
recording_topic = topic_root+"/recording"	
preds_topic = topic_root+"/predictions"

clientID = f"client_{args.model.split('/')[-1]}"
##################################################################################################


######### DEFINE SUBCRIPTION HANDLER #############################################################
class myHandler(MQTT_Handler):
	def notify(self, topic, msg):
		if topic == recording_topic:
			data = json.loads(msg)
			
			# logging
			print(f"client_{args.model.split('/')[-1]}: Received recording from {data['bn']}, recording id: {data['record_id']}")
			
			
			recording = np.frombuffer(base64.b64decode(data["e"][0]["vd"]), dtype=np.float32)		# take the recording from the message (base64 string ->  numpy object)
			recording = np.reshape(recording, (1, 49, 10, 1))				# reshape the numpy array
			
			interpreter.set_tensor(input_details[0]['index'], recording)	# pass the input data to the model
			interpreter.invoke()
			
			y_pred = interpreter.get_tensor(output_details[0]['index'])		# get the output of the last layer of the model 
			y_pred = y_pred.squeeze() 										#remove batch dim
			# also the logist are sent with SenML and are encoded in base64
			encoded_logits = base64.b64encode(y_pred).decode()				# encode the logits (numpy array -> base64 string)
			
			# send the message with SenML
			logits = {"logits": encoded_logits, "device_id": f"client_{args.model.split('/')[-1]}", "record_id":data["record_id"]}
			# record id and device_id are additional fields used for logging and debugging
			
			# logging
			print(f"client_{args.model.split('/')[-1]}: Responding to {data['bn']} with the inference of record {data['record_id']}")
			
			handler.myMqttClient.myPublish(preds_topic, json.dumps(logits), QOS)	# publish message on predictions topic
##################################################################################################


######## START MQTT CLIENT #######################################################################		
handler = myHandler(clientID)						    # init the handler
handler.run()										    # start the MQTT client
handler.myMqttClient.mySubscribe(recording_topic, QOS)		# subscribe to recording topic
##################################################################################################

while True:
	time.sleep(0.01)


