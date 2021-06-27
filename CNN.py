from keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.layers import Flatten

(train_X, train_y), (test_X, test_y) = mnist.load_data()
#Reshap mnist to be 3D To Fit the model slover
img_rows=train_X[0].shape[0]

img_cols=test_X[0].shape[1]
train_X=train_X.reshape(train_X.shape[0],img_rows,img_cols,1)

test_X=test_X.reshape(test_X.shape[0],img_rows,img_cols,1)
def training():
    model = models.Sequential()
    '''first layer of convlution size of 32 filters (33 mask )'''
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
    '''First pool (max) 22 mask'''
    model.add(layers.MaxPooling2D((2, 2)))
    ''''second layer of convlution size of  64 filters (3*3 mask ) 
    with linear activation function (relue) '''
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    '''thrid layer''' 
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    '''Convert the 3D Matrix to 1D Feature vector to fit in the Ann layer '''
    model.add(layers.Flatten())
    '''Ann layer'''
    model.add(layers.Dense(512, activation='relu'))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.save('E:/Year 4/Semster 2/Computer Vision/Assignments/CNN/my_model')
#training()
new_model = tf.keras.models.load_model('E:/Year 4/Semster 2/Computer Vision/Assignments/CNN/my_model')
test_loss, test_acc = new_model.evaluate(test_X,  test_y, verbose=2)
train_loss, train_acc = new_model.evaluate(train_X,  train_y, verbose=2)
print(test_acc)
print(train_acc)