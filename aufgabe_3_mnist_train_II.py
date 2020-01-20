from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from metrics import metrics
import numpy as np
import sklearn.datasets 
# from keras.models import Sequential, load_model
# from numpy import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import array as arr
import regex

batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Inhalt eines Ordners
anz_datensaetze_input = 0

# Daten einlesen für die Testmenge
test_index_dataset = 500
train_index_dataset = 10000

for aktuelle_zahl in range(0, 10):

    verzeichnis = './TESTSET/' + str(aktuelle_zahl) + '/'

    regex_png = regex.compile('[0-9a-zA-Z]*.png')


    with os.scandir(verzeichnis) as entries:

        #print("Anzahl der Einträge ", len(entries))
        for entry in entries:
            if regex.search(regex_png, entry.name) :
            
                # Einlesen der Datei
                pfad = verzeichnis + entry.name
                print("Diese Datei wird eingelesen", pfad)
                image = mpimg.imread(pfad)
                image *= 255
                
                # Bilder in den Testdatensatz überschreiben
                x_test[test_index_dataset] = image
                y_test[test_index_dataset] = aktuelle_zahl
                test_index_dataset=test_index_dataset+1

                # für Training - ab Index 10.000
                x_train[train_index_dataset] = image
                y_train[train_index_dataset] = aktuelle_zahl
                train_index_dataset = train_index_dataset + 1


print("So viele Datensätze wurden eingelesen ", test_index_dataset)




if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# Start des Lernprozesses
# der optimizer Parameter kann noch weitere Verbesserungen enthalten
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# Parameter für das Testing werden festgelgt
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Testing
    # Scalar test loss (if the model has a single output and no metrics)  
    #   or list of scalars (if the model has multiple outputs  
    #   and/or metrics). The attribute `model.metrics_names` will give you  
    #   the display labels for the scalar outputs.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Predicition
x_test_2 = []
#for a in range(0, 50):
 #   print(x_test[0][a])
    #x_test_2[i] = x_test[i]

pre_X = model.predict(x_test, batch_size=batch_size)

i = 0
w, h = 10, 10;
matrix = [[0 for x in range(w)] for y in range(h) ]

number_of_zeros = 0

for b in range(0, 1000):

    digit_array = pre_X[b]
    label_array = y_test[i]
    value_label = np.argmax(label_array)
    value_prediciton = np.argmax(digit_array)

    matrix[value_label][value_prediciton] = matrix[value_label][value_prediciton] + 1

    i = i + 1

    if value_label == 0:
        number_of_zeros = number_of_zeros + 1

    #print("label ", label_array)
    #print("value ", value_prediciton)


print("number of zeros ", number_of_zeros)
print("Testsample: ", i)
for y in matrix:
    print(y)
