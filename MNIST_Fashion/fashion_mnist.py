import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
requests.packages.urllib3.disable_warnings()
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

from tensorflow.keras.datasets import fashion_mnist

#scalig the values
from tensorflow import train
from tensorflow.keras.utils import to_categorical

# early stop and tencerboard
import datetime
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint

# actual tencerflow imports for cnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten

from sklearn.metrics import classification_report,confusion_matrix

earlystop = EarlyStopping(monitor='val_loss',patience=2)

log_dir = "logs/colour/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

board = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1
)

(x_train,y_train),(x_test,y_test)= fashion_mnist.load_data()
# show img
# a = plt.imshow(x_train[10])
# plt.show(a)


x_train = x_train/255
x_test = x_test/255

#batch_size,width,hight,colourchannel
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#converting them to catogorical values
y_cat_test = to_categorical(y_test,num_classes=10)
y_cat_train = to_categorical(y_train,num_classes=10)

model = Sequential()

#convolutional layer
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28,28,1), activation='relu'))
#poolong layer
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("started training")

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

#to run the model for the first time use this
# model.fit(x_train, y_cat_train, epochs=3, validation_data=(x_test, y_cat_test),
#           callbacks=[earlystop,board,cp_callback])

#to lode the already doen model use this command
model.load_weights(checkpoint_path)

predictions = model.predict_classes(x_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

single_image = x_test[10]

print(model.predict_classes(single_image.reshape(1,28,28,1)))




