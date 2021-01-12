import requests
import json
import base64
import numpy as np
from datetime import datetime
import tensorflow as tf

from scipy.stats import entropy

SAMPLING_RATE = 16000

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
        num_spectrogram_bins = (frame_length) // 2 + 1

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
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

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


#####################################################

COMM_COST = 0
URL = "http://192.168.1.170:8080"  # url del pc


def BigRequest(url, file_path):
    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")

    wav = tf.io.read_file(file_path).numpy()
    encoded_audio = base64.b64encode(wav).decode()

    json_audio = {"n": "audio", "u": "", "t": timestamp, "v": encoded_audio}
    out = {"bn": "little_service", "e": json_audio}

    try:
        r = requests.post(url=url, json=out)
    except requests.exceptions.RequestException as err:
        raise SystemExit(err)

    body = r.json()
    sample_label = body["e"]["label"]
    return sample_label, len(json.dumps(out))


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

test_ds = generator.make_dataset(test_files, False)

interpreter = tflite.Interpreter('./models/' + filename + '.tflite')

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

accuracy = 0
count = 0

for x, y_true in test_ds.unbatch().batch(1):
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])

    y_pred = y_pred.squeeze()  # remove batch dim

    print("output of the little model : " + str(y_pred) + "\n")

    y_predicted_value = np.argmax(y_pred)
    y_true = y_true.numpy().squeeze()

    accuracy += y_predicted_value == y_true  # 1 if True, 0 otherwise
    count += 1

    inference = y_pred

    if not SuccessChecker_BinEntropy(inference, 0.8):
        print("NO SUCCESS")
        label, cost = BigRequest(URL, audio_path)
        print("Prediction is : " + str(label) + "\n")
        if label != "ERROR":
            COMM_COST += cost
    else:
        print("SUCCESS. Prediction is " + str(y_predicted_value) + "\n")

accuracy /= float(count)

print("Accuracy {}".format(accuracy))

# ##################################################### Main di prova
#
# for i in range(30):
#     path = "raw_data/yes1.wav"
#     inference = np.array([0.5, 0.5])
#     if not SuccessChecker_BinEntropy(inference, 0.8):
#         label, cost = BigRequest(URL, path)
#         print(label)
#         if label != "ERROR":
#             COMM_COST += cost
#
# print("Communication cost: {}MB".format(np.round(COMM_COST / 1000000), 1))
