import sklearn
import pandas as pd
import numpy as np
from keras.src.layers import Dense, Dropout
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split
from keras import datasets, preprocessing, Sequential,layers,utils,optimizers


data=pd.read_csv("spam.csv",encoding="latin-1")
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data = data.rename(columns={'v1': 'target', 'v2': 'text'})
target = data.pop('target').values
text = data.pop('text').values
tokenizer =Tokenizer(num_words=5000)
tokenizer.fit_on_texts(text)
text = tokenizer.texts_to_sequences(text)
text = pad_sequences(text, maxlen=100)
target = to_categorical(target, num_classes=2)
x_train, x_test, y_train, y_test = train_test_split(text, target, test_size=0.2)
model=Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
