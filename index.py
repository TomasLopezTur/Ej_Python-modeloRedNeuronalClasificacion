import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

categories=['ananas','kiwis','manzanas']
images=[]
labels=[]

x=0
for i in categories:
    c=1
    for o in range(10):
        img=cv2.imread(f'./img/' + i +'/'+ str(c)+'.jpg',0 )
        img=cv2.resize(img , (200, 200))
        img=np.asarray(img)
        images.append(img)
        c += 1
        labels.append(x)
    x += 1

model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),input_shape=(200,200,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),input_shape=(200,200,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=100,activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

images = np.array(images)
labels = np.array(labels)
model.fit(images, labels, epochs = 30)

test=cv2.imread("./img/testII.jpg",0)
test=cv2.resize(test,(200,200))
test=np.asarray(test)
test=np.array([test])
result=model.predict(test)
print(result)

clase_predicha = np.argmax(result)
porcentaje_confianza = result[0][clase_predicha] * 100
print(f'Predicción: {categories[clase_predicha]} con un {porcentaje_confianza:.2f}% de confianza.')
print("result: " , categories[np.argmax(result[0])])