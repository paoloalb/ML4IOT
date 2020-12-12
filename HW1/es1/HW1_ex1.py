import argparse
import numpy as np
import tensorflow as tf
import time
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help="The source directory")
parser.add_argument('--output', type=str, help="The output TFRecord file")
args = parser.parse_args()

source_directory = args.input + "/"
output_file = args.output


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


with tf.io.TFRecordWriter(output_file) as writer:
    with open(source_directory + 'samples.csv') as input_file:
        reader = csv.reader(input_file, delimiter=',')
        n_records = 0
        for row in reader:
            n_records += 1

            POSIX_timestamp = np.int32(time.mktime(time.strptime(row[0] + " " + row[1], "%d/%m/%Y %H:%M:%S")))
            p_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[POSIX_timestamp]))

            t_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[np.int32(row[2])]))
            h_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[np.int32(row[3])]))

            wav = tf.io.read_file(source_directory + row[4])
            a_feature = _bytes_feature(wav)

            mapping = {'timestamp': p_feature,
                       'temperature': t_feature,
                       'humidity': h_feature,
                       'audio': a_feature}

            record = tf.train.Example(features=tf.train.Features(feature=mapping))
            writer.write(record.SerializeToString())

print("Number of records: {}".format(n_records))
print("Size of outputfile: {} bytes".format(os.path.getsize(output_file)))
