import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import sample

model_url="https://tfhub.dev/google/yamnet/1"
yamnet_model=hub.load(model_url)

test_file = tf.io.read_file('sample.wav')
test_audio, _ = tf.audio.decode_wav(contents=test_file)
print(test_audio.shape)

def load_audio(filename):
    binary_audio=tf.io.read_file('sample.wav')
    test_audio, rate = tf.audio.decode_wav(contents=binary_audio)
    return tf.reduce_mean(test_audio, axis=1), rate
path='sample.wav'
waveform,sample_rate=load_audio(path)
plt.plot(waveform.numpy())
plt.show()