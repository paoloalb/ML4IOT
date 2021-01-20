import requests
import json
import base64
import numpy as np
from datetime import datetime
import tensorflow as tf
import os
import tensorflow.lite as tflite
from SignalGenerator import SignalGenerator
from scipy.stats import entropy
os.environ["CUDA_VISIBLE_DEVICES"] = "" 					# ignore GPU devices
import time

SAMPLING_RATE = 16000

COMM_COST = 0
MAX_COMM_COST = 4.5*(2**20)

URL = "http://0.0.0.0:8080"  # url del pc


def SuccessChecker_BinEntropy(inf_array, threshold):
    print("Entropy: " + str(entropy(inf_array, base=2)))
    if entropy(inf_array, base=2) <= threshold:
        return True
    else:
        return False


def SuccessChecker_FirstSecond(inf_array, threshold):
    ord_array = np.sort(inf_array)[::-1]
    if (ord_array[0] - ord_array[1]) >= threshold:
        return True
    else:
        return False

def BigRequest(url, file_path, generator):
	dateTimeObj = datetime.now()
	timestamp = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
	
	binary = tf.io.read_file(file_path)
	int16_wav = np.frombuffer(binary.numpy(), dtype=np.int16)
    
	encoded_audio = base64.b64encode(int16_wav).decode()
	#encoded_audio = base64.b85encode(wav).decode()

	json_audio = {"n": "audio", "u": "/", "t": 0, "vd": encoded_audio}
	out = {"bn": "little_service", "bt": timestamp, "e": json_audio}
    
	if (COMM_COST+len(json.dumps(out)))>MAX_COMM_COST:
		return None, 0
	else:
		try:
			r = requests.post(url=url, json=out)
		except requests.exceptions.RequestException as err:
			raise SystemExit(err)

		body = r.json()
		sample_label = body["label"]
		return int(sample_label), len(json.dumps(out))


#### DATASET DOWNLOAD ############################################################################
zip_path = tf.keras.utils.get_file(
	origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
	fname="mini_speech_commands.zip",
	extract=True,
	cache_dir=".",
	cache_subdir="data")

data_dir = os.path.join(".", "data", "mini_speech_commands")
##################################################################################################


# Reads test set
test_files = []
with open("kws_test_split.txt", "r") as test_file:
    for filename in test_file:
        test_files.append(filename[:-1])

f = open("labels.txt", "r")
LABELS = f.read().split(" ")
f.close()

STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}

options = STFT_OPTIONS

generator = SignalGenerator(LABELS, SAMPLING_RATE, **options)

interpreter = tflite.Interpreter('models/Group7_little2.tflite')

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

accuracy = 0
new_accuracy = 0
count = 0

insucces_count = 0

entr = 0

big_right, small_right = 0, 0

total_time, inference_time = 0, 0

for n, path in enumerate(test_files):
	
    x, y_true, pr_time = generator.preprocess_with_stft(path)

    interpreter.set_tensor(input_details[0]['index'], [x])
    
    start_inf = time.time()
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    
    inference_time += end-start_inf
    total_time += end-start_inf+pr_time
	
	
    y_pred = y_pred.squeeze()  # remove batch dim
    y_pred = tf.nn.softmax(y_pred)

    y_predicted_value = np.argmax(y_pred)
    y_true = y_true.numpy().squeeze()

    accuracy += y_predicted_value == y_true  # 1 if True, 0 otherwise
    count += 1

    inference = y_pred

    entr += entropy(inference, base=2)

    if not SuccessChecker_FirstSecond(inference, 0.66):
        print("NO SUCCESS")
        insucces_count += 1
        label, cost = BigRequest(URL, test_files[n], generator)
        
        # if the function returns None, 0 it means that we are exceeding the communication limit
        # so the big client inference is skipped and the label is taken from the little model instead
        if label==None and cost==0:
        	new_accuracy += y_predicted_value == y_true  # 1 if True, 0 otherwise
        	print("Communication cost exceeded, the inference is limited to only little model now")
        	if y_predicted_value == y_true:
        		small_right += 1
       		else:
        		print(f"MISTAKE!: Predictions: {y_pred}")
        	
        else:
            new_accuracy += label == y_true  # 1 if True, 0 otherwise
            COMM_COST += cost
            print(f"Big model prediction: {label}, true value: {y_true}")
            if label == y_true:
            	big_right += 1	
       	
    else:
        new_accuracy += y_predicted_value == y_true  # 1 if True, 0 otherwise
        print("SUCCESS. Prediction is " + str(y_predicted_value) + "\n")
        if y_predicted_value == y_true:
        	small_right += 1
        
        
    print(f"recording #{n}: current accuracy: {new_accuracy/(n+1):.4f}")

accuracy /= float(count)

new_accuracy /= float(count)

#print("Accuracy {}".format(accuracy))
print("New accuracy {}".format(new_accuracy))

print("There were : " + str(insucces_count) +  "/" + str(count)  + " errors")

print(f"Partial small accuracy: {small_right/(count-insucces_count)}")
print(f"Partial big accuracy: {big_right/insucces_count}")

print("average entropy: " + str( entr / count ))
print(f"communication cost : {COMM_COST/(2**20):.5f} MB")

print(f"Average total time : {total_time*1000/count:.4f} ms")
print(f"Average inference time : {inference_time*1000/count:.4f} ms")



