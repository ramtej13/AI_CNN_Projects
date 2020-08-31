import matplotlib.pyplot as plt
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

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
x_train = x_train/255
x_test = x_test/255
single_image = x_test[10]
fig = plt.figure
plt.imshow(single_image)
plt.show()