import tensorflow as tf
from tensorflow import keras
(train_images,train_labels),(test_images,test_labels)=keras.datasets.mnist.load_data()
train_images=train_images.reshape((60000,28,28,1))
train_images=train_images/255.0
test_images=test_images.reshape((10000,28,28,1))
test_images=test_images/255.0
model=keras.Sequential(
    [keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
     keras.layers.MaxPooling2D(pool_size=(2, 2)),
     keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
     keras.layers.MaxPooling2D(pool_size=(2, 2)),
     keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
     keras.layers.Flatten(),
     keras.layers.Dense(64, activation='relu'),
     keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=5,batch_size=64,
          validation_data=(test_images,test_labels))
test_loss, test_acc = model.evaluate(test_images,test_labels)




