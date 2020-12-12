import argparse
import pyaudio
import time
import numpy as np
import tensorflow as tf
from scipy import signal
import os
import io
import sys
from subprocess import Popen

# suppress warnings
os.close(sys.stderr.fileno())

# PARSING ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--num-samples", type=int, default=5)
parser.add_argument("-output", type=str, default="./out")

args = parser.parse_args()

NUM_SAMPLES = args.num_samples
OUTPUT = args.output


# recording function
def record(stream, RATE, CHUNK, RECORD_SECONDS):
    # start recording
    stream.start_stream()

    with io.BytesIO() as data_buffer:

        # change profile to powersave before recording
        Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'], shell=True)

        # iteration in which the power policy is switched (nearly 80% of total time)
        switch_profile_iter = int(np.round(RATE / CHUNK * RECORD_SECONDS * 0.85))

        n_iter = int(RATE / CHUNK * RECORD_SECONDS)

        # number of chunks is equal to the total number of bits divided by the dimension of a chunk
        for i in range(0, n_iter):

            # reached the percentage of total recording, switch to performance
            if i == switch_profile_iter:
                Popen(['sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],
                      shell=True)

            data_buffer.write(stream.read(CHUNK, exception_on_overflow=False))

        # get data from buffer
        recording = np.frombuffer(data_buffer.getvalue(), dtype=np.int16)

    # stop Recording
    stream.stop_stream()

    return recording


# resampling function
def resample(recording, new_sampling_rate):
    resampled_audio = signal.resample_poly(recording, 1, RATE / new_sampling_rate)
    return resampled_audio


def spectrogram(resampled, frame_length, frame_step):
    # convert to tensor
    tf_audio = tf.constant(resampled, dtype=np.float32)

    # scale to [-1 1]
    normalized_tf_audio = tf.math.divide(
        tf.add(
            tf_audio,
            32768
        ),
        tf.constant(65535, dtype=float),
    )

    normalized_tf_audio = tf.math.subtract(normalized_tf_audio, 0.5)
    normalized_tf_audio = tf.multiply(normalized_tf_audio, 2)

    # compute stft and spectrogram
    stft = tf.signal.stft(normalized_tf_audio, frame_length=frame_length, frame_step=frame_step,
                          fft_length=frame_length)
    spectrogram = tf.abs(stft)

    return spectrogram


def mfcc(spectrogram, linear_to_mel_weight_matrix):
    mel_spectrogram = tf.tensordot(
        spectrogram,
        linear_to_mel_weight_matrix,
        1)
    mel_spectrogram.set_shape((49, 40))  # spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # compute mfcc fom mel spectrogram and take first 10 coefficients
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :10]
    return mfccs


# FIXED PARAMETERS
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 2400
RECORD_SECONDS = 1
RESAMPLING_RATE = 16000
prefix = "mfccs"

# instantiate PyAudio
audio = pyaudio.PyAudio()

# open connection
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK, start=False, input_device_index=0)

# pre compute everything that does not depend on the audio signal
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(40, 321, RESAMPLING_RATE, 20, 4000)
frame_length = int(RESAMPLING_RATE * 0.040)
frame_step = int(RESAMPLING_RATE * 0.020)

# reset cpu time stats
Popen(['sudo sh -c "echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"'], shell=True).wait()

# record n times
for s in range(NUM_SAMPLES):
    print(f"\nRecording number {s + 1}")
    tic1 = time.time()
    recording = record(stream, RATE, CHUNK, RECORD_SECONDS)
    tic2 = time.time()
    resampled = resample(recording, RESAMPLING_RATE)
    tic3 = time.time()
    spect = spectrogram(resampled, frame_length=frame_length, frame_step=frame_step)
    tic4 = time.time()
    mfccs = mfcc(spect, linear_to_mel_weight_matrix)
    tic5 = time.time()

    # write on disk
    byte_string = tf.io.serialize_tensor(mfccs)
    tf.io.write_file(OUTPUT + f"/{prefix}{s}.bin", byte_string)
    toc = time.time()

    print(f"Recording: \t{tic2 - tic1:.3f} s")
    print(f"Resampling: \t{tic3 - tic2:.3f} s")
    print(f"Spectrogram: \t{tic4 - tic3:.3f} s")
    print(f"MFCCS: \t\t{tic5 - tic4:.3f} s")
    print(f"Writing: \t{toc - tic5:.3f} s")

    print(f"\tElapsed: {toc - tic1:.3f} s")
    print(f"\tPreprocessing: {toc - tic2:.3f} s")

# close connection
stream.close()
audio.terminate()

print("\n")
# show cpu time stats
Popen(['cat /sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state'], shell=True)

# reset into powersave
Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'], shell=True)
