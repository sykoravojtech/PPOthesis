import tensorflow as tf
import os
from tensorflow.python.client import device_lib

print(tf.__version__)
print([device.name for device in device_lib.list_local_devices() if device.name != None])