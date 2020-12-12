import tensorflow as tf
import numpy as np


class WindowGenerator:
    def __init__(self, input_width, mean, std):
        self.input_width = input_width
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2]) # one for tem and one for hum
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2]) # other dimension to match data shape

    def split_window(self, features):
        inputs = features[:, :-6, :]

        labels = features[:, -6:, :]
        num_labels = 2


        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, 6, num_labels])


        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.input_width+6,
                sequence_stride=1,
                batch_size=32)

        # for i in ds:
        #     print(i.shape)  # (32, 12, 2)

        ds = ds.map(self.preprocess)

        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

