import tensorflow as tf
import os
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow.python.client import device_lib

print(f"TensorFlow version = {tf.__version__}")
print(f"Devices available: {[device.name for device in device_lib.list_local_devices() if device.name != None]}")
