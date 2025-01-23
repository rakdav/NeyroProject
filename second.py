import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import sample

from first import class_names

model_url="https://tfhub.dev/google/yamnet/1"
yamnet_model=hub.load(model_url)

def load_audio(filename):
    binary_audio=tf.io.read_file(filename)
    test_audio, rate = tf.audio.decode_wav(contents=binary_audio)
    return tf.reduce_mean(test_audio, axis=1), rate
path='qwe.wav'
waveform,sample_rate=load_audio(path)
print(f"Sample rate: {sample_rate.numpy()} Hz")
print(f"Waveform shape: {waveform.shape}")
plt.plot(waveform.numpy())
plt.show()

class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
class_names = np.array([name for  name in open(class_map_path).read().splitlines()])
print(class_names)
def predict(audio):
    score,embeddings,spectrogram = yamnet_model(audio)
    scores_np=score.numpy()
    return scores_np

scores=predict(waveform)
predict_class=class_names[scores.mean(axis=0).argmax()]
print(f"Predict class: {predict_class}")
