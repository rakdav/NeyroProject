#CNN
import tensorflow as tf
import keras
from keras import datasets, layers,utils,models
import numpy as np
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from PIL import Image
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train=np.reshape(x_train,(60000,28,28,1))/255
x_test=np.reshape(x_test,(10000,28,28,1))/255

y_train=keras.utils.to_categorical(y_train)
y_test=keras.utils.to_categorical(y_test)

#Добавляем сверхточные слои
model=keras.models.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#Добавляем слои полносвяОбучим нашу модель на наборе данных MNISTзной нейронной сети
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Обучим нашу модель на наборе данных MNIST
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

def predict_image(image_path):
    img=Image.open(image_path).resize((28,28)).convert('L')
    img_array=np.array(img)
    plt.imshow(img_array)
    plt.show()
    res=model.predict(img_array.reshape(1,28,28,1)/255.0)[0]
    print (np.argmax(res))

predict_image("two.png")