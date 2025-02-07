import tensorflow as tf
from keras import layers, datasets, Sequential
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

(x_train,y_train),(x_test,y_test)=datasets.mnist.load_data()
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(x_train[i],cmap='gray')
    plt.axis('off')
plt.show()
x_train=x_train.reshape(-1,28,28,1).astype('float32')/255.0
x_test=x_test.reshape(-1,28,28,1).astype('float32')/255.0
model=Sequential(
    [
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax'),
    ]
)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

def predict_image(image_path):
    img=Image.open(image_path).resize((28,28)).convert('L')
    img_array=np.array(img)
    plt.imshow(img_array)
    plt.show()
    res=model.predict(img_array.reshape(1,28,28,1)/255.0)[0]
    print (np.argmax(res))

predict_image("two.png")
