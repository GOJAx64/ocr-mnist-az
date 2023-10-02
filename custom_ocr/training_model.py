import tensorflow
import numpy as np
import zipfile
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

'''
MNIST 
datasets de training y test, con sus respectivas etiquetas
El dataset de entrenamiento cuenta con 60,000 imagenes y el de test con 10,000 ambos con imagenes de 28x28
((60000, 28, 28), (10000, 28, 28))
'''
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

digits_data = np.vstack([train_data, test_data])
digits_labels = np.hstack([train_labels, test_labels])

'''
A-Z 
descargamos el dataset para las letras entre A-Z
El dataset cuenta con 372450 imagenes, aplicamos un reshape de 28x28
'''
# wget https://iaexpert.academy/arquivos/alfabeto_A-Z.zip
zip_object = zipfile.ZipFile(file='/content/alfabeto_A-Z.zip', mode='r')
zip_object.extractall('./')
zip_object.close()

dataset_az = pd.read_csv('/content/A_Z Handwritten Data.csv').astype('float32')
alphabet_data = dataset_az.drop('0', axis=1)
alphabet_labels = dataset_az['0']
alphabet_data = np.reshape(alphabet_data.values, (alphabet_data.shape[0], 28, 28))  # (372450, 28, 28)

'''
Uniendo Datasets
Representamos las etiquetas unicas de los numeros, es decir, del 0 al 9
Representamos las eqtiquetas de las letras del alfabeto ingles, A-Z
'''
alphabet_labels += 10  # Sumamos 10 para representar a los números
data = np.vstack([alphabet_data, digits_data])  # Unimos los datasets
labels = np.hstack([alphabet_labels, digits_labels])  # (data, labels): ((442450, 28, 28), (442450,))
data = np.array(data, dtype='float32')
data = np.expand_dims(data, axis=-1)

'''
Preprocesamiento de los datos
Normalizamos los datos poniendolos dentro del rango 0-1, debido a que si dejamos los valores iniciales (0-255) 
esto generara numeros mucho mas grandes y tomara mas tiempo. esto lo hacemos dividiendo la data entre 255.
'''
data /= 255.0

'''
Usaremos la funcion softmax, esta devolvera una probabilidad, la probabilidad mas cercana a un numero en concreto es 
la label que retornaremos. Pero para esto transformamos los valores a una nueva representacion con LabelBinarizer() 
que nos da una representacion llamada OneHotEncoder
'''
le = LabelBinarizer()
labels = le.fit_transform(labels)

'''Creamos los pesos de las clases'''
classes_total = labels.sum(axis=0)
classes_weights = {}
for i in range(0, len(classes_total)):
    classes_weights[i] = classes_total.max() / classes_total[i]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1, stratify=labels)
augmentation = ImageDataGenerator(rotation_range=10, zoom_range=0.05, width_shift_range=0.1, height_shift_range=0.1,
                                  horizontal_flip=False)

'''Red Neuronal convolucional'''
network = Sequential()

# capa 1
# cada filtro hara multiplicacion de matrices con cada uno de los pixeles de la imagen
# kernel_size es el tamaño de la matriz,          funcion de activación, tamaño de la entrada (imagen)
network.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(MaxPool2D(pool_size=(2, 2)))

# capa 2
network.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
network.add(MaxPool2D(pool_size=(2, 2)))

# capa 3
network.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
network.add(MaxPool2D(pool_size=(2, 2)))

network.add(Flatten())  # convierte la matriz en vector

#neuronas
network.add(Dense(64, activation='relu'))
network.add(Dense(128, activation='relu'))
network.add(Dense(36, activation='softmax'))

network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
network.summary()

name_labels = '0123456789'
name_labels += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
name_labels = [l for l in name_labels]
file_model = 'custom_ocr.model'
epochs = 20
batch_size = 128
checkpointer = ModelCheckpoint(file_model, monitor='val_loss', verbose=1, save_best_only=True)

history = network.fit(augmentation.flow(X_train, y_train, batch_size=batch_size),
                      validation_data=(X_test, y_test),
                      steps_per_epoch=len(X_train) // batch_size, epochs=epochs,
                      class_weight=classes_weights, verbose=1, callbacks=[checkpointer])

'''
Evaluando la Red Neuronal
Usamos el set de test para que haga las predicciones de todos y podamos acceder a las mismas
'''
predictions = network.predict(X_test, batch_size=batch_size)
prediction = np.argmax(predictions[0])
print(name_labels[prediction])
label_pred = np.argmax(y_test[0])
print(name_labels[np.argmax(label_pred)])

network.evaluate(X_test, y_test)
# 2766/2766 [==============================] - 40s 14ms/step - loss: 0.1537 - accuracy: 0.9402
# [0.15367434918880463, 0.9402192234992981] [ERROR, ACCURACY]
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=name_labels))
print(history.history.keys())  # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


'''Guardando el modelo'''
# network.save('network', save_format='h5')


'''
PROBANDO EL MODELO
import cv2

loaded_network = load_model('network')
loaded_network.summary()
img = cv2.imread('/test/example.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
value, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
img = cv2.resize(thresh, (28, 28))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis = -1)
img = np.reshape(img, (1,28,28,1))
prediction = loaded_network.predict(img)
index = np.argmax(prediction)
name_labels[index]
'''
