#!/bin/bash

python3 inference_client.py --model models/0.tflite --qos 0 &
python3 inference_client.py --model models/1.tflite --qos 0 &
python3 inference_client.py --model models/2.tflite --qos 0 &
python3 inference_client.py --model models/3.tflite --qos 0 &


