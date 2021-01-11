import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import zlib  # to compress the model
import tensorflow.lite as tflite
import tensorflow_model_optimization as tfmot  # for magnitude based pruning

parser = argparse.ArgumentParser()
parser.add_argument("--version",
                    type=str,
                    # required=True,
                    default="little",  # togliere il default alla consegna
                    help="Version of the model (big or little)")

args = parser.parse_args()

assert args.version in ["big", "little"], "Error: parameter version is not correct"


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


##################################################################################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Do not print info and warning messages
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ignore GPU devices

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname="mini_speech_commands.zip",
    extract=True,
    cache_dir=".",
    cache_subdir="data")

data_dir = os.path.join(".", "data", "mini_speech_commands")

train_files, val_files, test_files = [], [], []

with open("kws_train_split.txt", "r") as train_file:
    for filename in train_file:
        train_files.append(filename[:-1])

with open("kws_val_split.txt", "r") as val_file:
    for filename in val_file:
        val_files.append(filename[:-1])

with open("kws_test_split.txt", "r") as test_file:
    for filename in test_file:
        test_files.append(filename[:-1])

print(f"Train set size: {len(train_files)}")
print(f"Val set size: {len(val_files)}")
print(f"Test set size: {len(test_files)}")

f = open("labels.txt", "r")
LABELS = f.read().split(" ")
f.close()


def bigtraining():
    EPOCHS = 20
    LEARNING_RATE = 0.01
    SAMPLING_RATE = 16000
    options = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
               'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
               'num_coefficients': 10}
    strides = [2, 1]
    input_shape = [49, 10, 1]
    generator = SignalGenerator(LABELS, SAMPLING_RATE, **options)
    train_ds = generator.make_dataset(train_files, True)
    val_ds = generator.make_dataset(val_files, False)
    test_ds = generator.make_dataset(test_files, False)

    # CNN:
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=input_shape, filters=int(128), kernel_size=[3, 3], strides=strides,
                            use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=int(128), kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=int(128), kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=len(LABELS))
    ])

    filename = f"Group7_{args.version}"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./weights/' + filename,
        monitor='val_sparse_categorical_accuracy',
        patience=0,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch',
    )
    callbacks = [checkpoint]

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                  )

    model.summary()

    model.fit(train_ds, epochs=EPOCHS, verbose=1, validation_data=val_ds, callbacks=callbacks)
    model.load_weights('./weights/' + filename)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    if not os.path.exists("./models/"):
        os.makedirs("./models/")

    with open("./models/" + filename + ".tflite", 'wb') as fp:
        fp.write(tflite_model)
    print(f"Final model size: {os.path.getsize('./models/' + filename + '.tflite')} bytes")

    with open("./models/" + filename + ".zlib", 'wb') as fp:
        tflite_compressed = zlib.compress(tflite_model)
        fp.write(tflite_compressed)
    print(f"Compressed size: {os.path.getsize('./models/' + filename + '.zlib')} bytes")

    #### EVALUATE MODEL ON TEST SET ##################################################################
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
        y_pred = np.argmax(y_pred)

        y_true = y_true.numpy().squeeze()

        accuracy += y_pred == y_true  # 1 if True, 0 otherwise
        count += 1

    accuracy /= float(count)

    print("Accuracy {}".format(accuracy))
    ##################################################################################################


def littletraining():
    EPOCHS = 30
    LEARNING_RATE = 0.01
    STRUCTURED_W = 0.4# alpha
    MAGNITUDE_FS = 0.8  # final sparsity

    STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}

    SAMPLING_RATE = 16000


    options = STFT_OPTIONS
    strides = [2, 2]
    input_shape = [32, 32, 1]

    generator = SignalGenerator(LABELS, SAMPLING_RATE, **options)
    train_ds = generator.make_dataset(train_files, True)
    val_ds = generator.make_dataset(val_files, False)
    test_ds = generator.make_dataset(test_files, False)


    ALPHA = STRUCTURED_W


    # DS-CNN
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * ALPHA), kernel_size=[3, 3], strides=strides,
                            use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=int(256 * ALPHA), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=int(256 * ALPHA), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=len(LABELS))
    ])

    # filepath_base = f"./models/{args.version}"
    filename = f"Group7_{args.version}"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./weights/' + filename,
        monitor='val_sparse_categorical_accuracy',
        patience=0,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch',
    )
    callbacks = [checkpoint]

    # MAGNITUDE BASED PRUNING
    pruning_params = {'pruning_schedule':
        tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30,
            final_sparsity=MAGNITUDE_FS,
            begin_step=len(train_ds) * 5,
            end_step=len(train_ds) * 15
        )
    }

    callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    model = prune_low_magnitude(model, **pruning_params)

    model.build([32, 32, 32])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                  )

    model.summary()

    model.fit(train_ds, epochs=EPOCHS, verbose=1, validation_data=val_ds, callbacks=callbacks)
    model.load_weights('./weights/' + filename)

    #### CONVERT TO TFLITE ############
    model = tfmot.sparsity.keras.strip_pruning(model)

    # # POST TRAINING QUANTIZATION
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Quantization:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # ##################################################################################################
    #
    if not os.path.exists("./models/"):
        os.makedirs("./models/")

    with open("./models/" + filename + ".tflite", 'wb') as fp:
        fp.write(tflite_model)
    print(f"Final model size: {os.path.getsize('./models/' + filename + '.tflite')} bytes")

    with open("./models/" + filename + ".zlib", 'wb') as fp:
        tflite_compressed = zlib.compress(tflite_model)
        fp.write(tflite_compressed)
    print(f"Compressed size: {os.path.getsize('./models/' + filename + '.zlib')} bytes")
    ##################################################################################################

    #### EVALUATE MODEL ON TEST SET ##################################################################
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
        y_pred = np.argmax(y_pred)

        y_true = y_true.numpy().squeeze()

        accuracy += y_pred == y_true  # 1 if True, 0 otherwise
        count += 1

    accuracy /= float(count)

    print("Accuracy {}".format(accuracy))
    ##################################################################################################

    print(
        "epochs: " + str(EPOCHS) +
        "\nlearning rate: " + str(LEARNING_RATE) +
        "\nalpha: " + str(STRUCTURED_W) +
        "\nMAGNITUDE_FS: " + str(MAGNITUDE_FS)
    )


if args.version == "big":
    bigtraining()
else:
    littletraining()


