import tf_keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import keras_tuner


################################################################################################################
## Paths and Constants

N=20
m_features = 12

my_data = np.genfromtxt('ML-CUP24-TR.csv', delimiter=',')
X=my_data[:, 1:13]
Y=my_data[:, 13:16]
print(X.shape)
print(Y.shape)
from tf_keras.layers import Input, Dense
from tf_keras.models import Model
from tf_keras.optimizers.legacy import SGD
from sklearn.model_selection import train_test_split
from tf_keras.metrics import MeanSquaredError
train_ratio = 0.60
validation_ratio = 0.20
test_ratio = 0.20

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
def regressor_model(m_features, regularizer=None, learning_rate=0.01):
    
    input_x = Input(shape = (m_features,))
    lin_fn1 = Dense(3, activation = None, kernel_regularizer = regularizer)(input_x)
    yx_model = Model(inputs = input_x, outputs = lin_fn1)
    yx_model.compile(loss = 'mean_squared_error', optimizer='sgd')
    return yx_model

reg_par = [0, 0.001, 0.01, 0.1, 1]
for i_reg in reg_par:
    start = time.time()
    yx_model = regressor_model(m_features,learning_rate=0.1,regularizer=tf_keras.regularizers.l2(i_reg))
    log_train = yx_model.fit(x_train, y_train, epochs = 100, batch_size = N, validation_data=(x_val, y_val))
    plt.plot(log_train.history['loss'])
    plt.plot(log_train.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

'''score = yx_model.evaluate(x_test, y_test)
    print(score)'''
'''def build_model(hp):
  model = tf_keras.Sequential()
  reg = hp.Choice('regularizer', [0.0, 0.001, 0.01, 0.1, 1.0])
  model.add(tf_keras.layers.Input(shape = (m_features,)))
  model.add(tf_keras.layers.Dense(3, kernel_regularizer= tf_keras.regularizers.l2(reg), activation= None))
  model.compile(loss='mean_squared_error', optimizer = SGD(hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")))
  return model

tuner = keras_tuner.RandomSearch(
    build_model,
    max_trials=10,
    # Do not resume the previous search in the same directory.
    overwrite=True,
    objective="val_accuracy",
    # Set a directory to store the intermediate results.
    directory="/tmp/tb",
)

tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]'''
