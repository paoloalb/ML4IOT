#!/bin/bash

python3 inference_client.py --model models/0.tflite &
python3 inference_client.py --model models/1.tflite &
python3 inference_client.py --model models/2.tflite &
python3 inference_client.py --model models/3.tflite &


