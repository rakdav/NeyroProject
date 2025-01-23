import tensorflow as tf
from tensorflow.keras import datasets,layers,models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()
train_images=train_images/255.0
test_images=test_images/255.0
class_names = ['автомобиль', 'самолет', 'собака', 'кошка', 'олень','свинина','корабль']
model=models.Sequential(
    [layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)),
     layers.MaxPooling2D(pool_size=(2, 2)),
     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
     layers.MaxPooling2D(pool_size=(2, 2)),
     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
     layers.Flatten(),
     layers.Dense(64, activation='relu'),
     layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=5,batch_size=32,validation_split=0.2)

def predict_image(image_path):
    img=Image.open(image_path).resize((32,32)).convert('RGB')
    img_array=np.array(img)/255.0
    plt.imshow(img_array,cmap=plt.cm.binary)
    plt.show()
    predict=model.predict(img_array.reshape(1,32,32,3))
    predicted_class=class_names[np.argmax(predict)]
    print('Predict:',predicted_class)

predict_image('6447967656.jpg')

