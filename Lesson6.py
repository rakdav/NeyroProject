from cProfile import label

import pandas as pd
import numpy as np
from keras import datasets, preprocessing, Sequential,layers
from keras.src.layers import LSTM, Dropout, Dense

data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")

data["Month"]=pd.to_datetime(data["Month"])
data=data.set_index(["Month"])
data=data.astype('float32')

train_data=data.iloc[:100,:]
test_data=data.iloc[100:,:]

def create_sequences(data,sequence_length):
    X = []
    y = []
    for i in range(len(data)-sequence_length-1):
        seq=data[i:(i+sequence_length),:]
        X.append(seq)
        y.append(data[i+sequence_length,:])
    return  np.array(X),np.array(y)

sequence_length = 12
# Создание последовательностей для тренировочного набора данных
x_train, y_train = create_sequences(train_data.values, sequence_length)
# Создание последовательностей для тестового набора данных
x_test, y_test = create_sequences(test_data.values, sequence_length)
model=Sequential()
model.add(LSTM(128,input_shape=(sequence_length,1),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=32,verbose=2)
train_predict=model.predict(x_train)
test_predict=model.predict(x_test)
train_predict=np.concatenate((train_predict,np.zeros((len(train_data)-len(train_predict),1))))
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(data.values,label="Исходные данные")
plt.plot(train_predict,label="Прогноз тренировочных данных")
plt.plot(test_predict,label="Прогноз тестовых данных")
plt.legend()
plt.show()
