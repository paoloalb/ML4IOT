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
args = parser.parse_args()
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
			
			print(f"client_{args.model.split('/')[-1]}: Received recording from {data['bn']}, recording id: {data['record_id']}")
			
			recording = np.frombuffer(base64.b64decode(bytes(data["e"][0]["vd"], "utf-8")), dtype=np.float32)
			recording = np.reshape(recording, (1, 49, 10, 1))
			interpreter.set_tensor(input_details[0]['index'], recording)
			interpreter.invoke()
			y_pred = interpreter.get_tensor(output_details[0]['index'])
		
			y_pred = y_pred.squeeze() #remove batch dim
			
			encoded_logits = base64.b64encode(y_pred).decode()
			logits = {"n": "logits", "t":int(time.time()), "vd": encoded_logits}
			message = {"bn": f"client_{args.model.split('/')[-1]}", "record_id":data["record_id"], "e": [logits]}
			
			print(f"client_{args.model.split('/')[-1]}: Responding to {data['bn']} with the inference of record {data['record_id']}")
			handler.myMqttClient.myPublish(preds_topic, json.dumps(message))
##################################################################################################


######## START MQTT CLIENT #######################################################################		
handler = myHandler(clientID)						    # init the handler
handler.run()										    # start the MQTT client
handler.myMqttClient.mySubscribe(recording_topic)		# subscribe to recording topic
##################################################################################################

while True:
	time.sleep(0.001)



'''
from MyMQTT import MyMQTT
from MQTT_Handler import MQTT_Handler
import time


if __name__ == "__main__":
	test = MQTT_Handler("subscriber 1")
	test.run()
	test.myMqttClient.mySubscribe("276545/recording")

	a = 0
	while (a < 30):
		a += 1
		time.sleep(1)

	test.end()

'''
