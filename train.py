import numpy as np
from keras.models import Sequential
from keras.layers import Dense



my_data = np.genfromtxt('anchors.csv', delimiter=',')


X = my_data[:,:-1]
y = my_data[:,-1]


# define the keras model
model = Sequential()
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=10, steps_per_epoch=4, batch_size=2)



_, accuracy = model.evaluate(X, y)


with open("metrics.csv","w") as outfile:
    outfile.write("accuracy: " + str(accuracy))

