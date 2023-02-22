"""
This file is meant for extracting data from the tensorboard file 
and plotting it in a more visually correct way
"""
import os
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8')

CB91_Purple = '#9D2EC5'
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
color_list = [CB91_Purple, CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

# Load Tensorboard data
path = "./data.v2"
data = tf.compat.v1.train.summary_iterator(path)
# Extract values
steps = []
values = []

for e in data:
    for v in e.summary.value:
        if v.tag == 'average score':
            # print(tf.make_ndarray(v.tensor))
            steps.append(e.step)
            values.append(float(tf.make_ndarray(v.tensor)))
            # print(f"{e=}\n{e.step=}\n{e.summary.value=}")
            # exit()

error = np.random.normal(1, 20, size=len(values))
print(f"{steps=}\n\n{values=}\n\n{error=}")

# Plot values
plt.plot(steps, values)

# https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars
plt.fill_between(steps, values-error, values+error)

plt.title('My title')
plt.xlabel('steps')
plt.ylabel('Average Score')

# plt.show()
plt.savefig("plot2.png")


