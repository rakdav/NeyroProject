#RNN

import nltk
from keras import datasets, preprocessing, Sequential
from keras.src.layers import Embedding, LSTM, Dense
from keras.src.utils import pad_sequences
from nltk import pad_sequence

from Lesson4 import x_train, y_train, x_test, y_test, model, test_loss

vocab_size=20000
sequence_length=100

(x_train,y_train),(x_test,y_test)=datasets.imdb.load_data(num_words=vocab_size)
x_train=pad_sequences(x_train,maxlen=sequence_length)
x_test=pad_sequences(x_test,maxlen=sequence_length)
model = Sequential()
model.add(Embedding(vocab_size,128,input_length=sequence_length))
model.add(LSTM(64,return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=128,validation_data=(x_test,y_test))

test_loss, test_acc=model.evaluate(x_test,y_test)
print("Test accuracy:", test_acc)




