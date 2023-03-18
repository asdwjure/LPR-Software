#!/bin/sh
#
# Run as sudo!
# User must first create a virtual environment. Then run this script.
# Install all packages for LPR-Software on Raspberry Pi
#
# Author: Jure Rebernik
# Magistrsko delo


pip install --upgrade pip
pip install opencv-python==4.5.4.60
pip install flask
pip install flask_sqlalchemy
pip install pytesseract
pip install matplotlib
pip install tflite_runtime
sudo apt install tesseract-ocr -y
sudo apt install libgl1 -y # OpenCV dependency