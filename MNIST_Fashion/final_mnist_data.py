import os
import requests
requests.packages.urllib3.disable_warnings()
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from tensorflow.keras.datasets import fashion_mnist


from tensorflow.keras.utils import to_categorical

import datetime
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten


earlystop = EarlyStopping(monitor='val_loss',patience=2)

log_dir = "logs/colour/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

board = TensorBoard(log_dir=log_dir,histogram_freq=1,write_graph=True,write_images=True,update_freq='epoch',profile_batch=2,embeddings_freq=1)

(x_train,y_train),(x_test,y_test)= fashion_mnist.load_data()
x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

y_cat_test = to_categorical(y_test,num_classes=10)
y_cat_train = to_categorical(y_train,num_classes=10)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28,28,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

userinput = int(input("type here:"))
if userinput == 1:
    model.fit(x_train, y_cat_train, epochs=3, validation_data=(x_test, y_cat_test),
              callbacks=[earlystop,cp_callback])
elif userinput ==2:
    model.load_weights(checkpoint_path)
elif userinput == 3:
    model.fit(x_train, y_cat_train, epochs=3, validation_data=(x_test, y_cat_test),
              callbacks=[earlystop,board,cp_callback])
else:
    print("choose correction option ")

single_image = x_test[10]
print("this is the item predicted from selection ",model.predict_classes(single_image.reshape(1,28,28,1)))



