import tensorflow as tf
import os
from tensorflow.python.client import device_lib

print(f"TensorFlow version = {tf.__version__}")
print(f"Devices available: {[device.name for device in device_lib.list_local_devices() if device.name != None]}")

# d = dict(p1=1, p2=2)
# d2 = dict(p1=1)
# d3 = dict()

# def f2(p1 = 0, p2 = 0):
#     print(p1, p2)
# f2(**d)
# f2(**d2)
# f2(**d3)