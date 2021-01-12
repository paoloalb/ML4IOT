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

SAMPLING_RATE = 16000

COMM_COST = 0
URL = "http://0.0.0.0:8080"  # url del pc

def SuccessChecker_BinEntropy(inf_array, threshold):
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

    wav, _ = generator.read(file_path)
    encoded_audio = base64.b64encode(wav).decode()

    json_audio = {"n": "audio", "u": "", "t": timestamp, "v": encoded_audio}
    out = {"bn": "little_service", "e": json_audio}
    
    
    try:
        r = requests.post(url=url, json=out)
    except requests.exceptions.RequestException as err:
        raise SystemExit(err)

    body = r.json()
    sample_label = body["e"]["label"]
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

#test_ds = generator.make_dataset(test_files, False)

interpreter = tflite.Interpreter('models/Group7_little.tflite')

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

accuracy = 0
new_accuracy = 0
count = 0

insucces_count = 0

entr = 0

big_right, small_right = 0, 0

for n, path in enumerate(test_files):
    x, y_true = generator.preprocess_with_stft(path)

    interpreter.set_tensor(input_details[0]['index'], [x])
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])

    y_pred = y_pred.squeeze()  # remove batch dim
    y_pred = tf.nn.softmax(y_pred)

    y_predicted_value = np.argmax(y_pred)
    y_true = y_true.numpy().squeeze()

    accuracy += y_predicted_value == y_true  # 1 if True, 0 otherwise
    count += 1

    inference = y_pred

    entr += entropy(inference, base=2)

    if not SuccessChecker_FirstSecond(inference, 0.25):
        print("NO SUCCESS")
        insucces_count += 1
        label, cost = BigRequest(URL, test_files[n], generator)
        #print("Prediction is : " + str(label) + "\n")
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
        
        
    print(f"recording #{n}: current accuracy: {new_accuracy/(n+1):.3f}")

accuracy /= float(count)

new_accuracy /= float(count)

print("Accuracy {}".format(accuracy))
print("New accuracy {}".format(new_accuracy))

print("There were : " + str(insucces_count) +  "/" + str(count)  + " errors")

print(f"Partial small accuracy: {small_right/(count-insucces_count)}")
print(f"Partial big accuracy: {big_right/insucces_count}")

print("average entropy: " + str( entr / count ))
print(f"communication cost : {COMM_COST/(2**20):.2f} MB")

