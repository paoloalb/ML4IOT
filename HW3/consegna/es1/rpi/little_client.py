import requests
import json
import base64
import numpy as np
from datetime import datetime
import tensorflow as tf
import os
import tensorflow.lite as tflite

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ignore GPU devices
import time

SAMPLING_RATE = 16000

COMM_COST = 0
MAX_COMM_COST = 4.5 * (2 ** 20)

URL = "http://0.0.0.0:8080"


#### SIGNAL GENERATOR CLASS ######################################################################
class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
                 num_mel_bins=None, lower_frequency=None, upper_frequency=None,
                 num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = frame_length // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                              frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                                       self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        tic = time.time()
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label, time.time() - tic

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


def SuccessChecker_FirstSecond(inf_array, threshold):
    ord_array = np.sort(inf_array)[::-1]
    if (ord_array[0] - ord_array[1]) >= threshold:
        return True
    else:
        return False


def BigRequest(url, file_path):
    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")

    binary = tf.io.read_file(file_path)
    int16_wav = np.frombuffer(binary.numpy(), dtype=np.int16)

    encoded_audio = base64.b64encode(int16_wav).decode()

    json_audio = {"n": "audio", "u": "/", "t": 0, "vd": encoded_audio}
    out = {"bn": "little_service", "bt": timestamp, "e": json_audio}

    if (COMM_COST + len(json.dumps(out))) > MAX_COMM_COST:
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

interpreter = tflite.Interpreter('Group7_little.tflite')

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

accuracy = 0
new_accuracy = 0
count = 0

insucces_count = 0

big_right, small_right = 0, 0

total_time, inference_time = 0, 0

for n, path in enumerate(test_files):

    x, y_true, pr_time = generator.preprocess_with_stft(path)

    interpreter.set_tensor(input_details[0]['index'], [x])

    start_inf = time.time()
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()

    inference_time += end - start_inf
    total_time += end - start_inf + pr_time

    y_pred = y_pred.squeeze()  # remove batch dim
    y_pred = tf.nn.softmax(y_pred)

    y_predicted_value = np.argmax(y_pred)
    y_true = y_true.numpy().squeeze()

    accuracy += y_predicted_value == y_true  # 1 if True, 0 otherwise
    count += 1

    inference = y_pred

    if not SuccessChecker_FirstSecond(inference, 0.66):
        print("NO SUCCESS")
        insucces_count += 1
        label, cost = BigRequest(URL, test_files[n])

        # if the function returns None, 0 it means that we are exceeding the communication limit
        # so the big client inference is skipped and the label is taken from the little model instead
        if label == None and cost == 0:
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

    print(f"recording #{n}: current accuracy: {new_accuracy / (n + 1):.4f}")

accuracy /= float(count)

new_accuracy /= float(count)


print("Big service was called " + str(insucces_count) + "/" + str(count) + " times")

print(f"Partial small accuracy: {100*small_right / (count - insucces_count):.3f} % ")
print(f"Partial big accuracy: {100*big_right / insucces_count:.3f} %")

print("\n\nAccuracy: {} %".format(new_accuracy*100))
print(f"Communication cost: {COMM_COST / (2 ** 20):.5f} MB")
