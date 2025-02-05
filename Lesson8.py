import tensorflow as tf
from keras import optimizers, Sequential
from keras.src.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(100,)),
    Dense(1, activation='sigmoid')
])
optimizer=optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])