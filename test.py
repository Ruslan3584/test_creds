import numpy as np
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import tensorflow as tf


my_data = np.genfromtxt('test.csv', delimiter=',')

model = tf.keras.models.load_model('saved_model/')

X = my_data[:,:-1]
y = my_data[:,-1]



_, accuracy = model.evaluate(X, y)


with open("test_metrics.csv","w") as outfile:
    outfile.write("accuracy: " + str(accuracy))

