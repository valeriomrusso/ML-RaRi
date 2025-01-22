import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import keras_tuner
from sklearn.preprocessing import StandardScaler


################################################################################################################
## Paths and Constants

m_features = 12

my_data = np.genfromtxt('ML-CUP24-TR.csv', delimiter=',')
X=my_data[:, 1:13]
Y=my_data[:, 13:16]
print(X.shape)
print(Y.shape)
from keras import Model
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras import metrics
train_ratio = 0.60
validation_ratio = 0.20
test_ratio = 0.20

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

scalerX = StandardScaler().fit(x_train)
scalerY = StandardScaler().fit(y_train)

x_train = scalerX.transform(x_train)
x_val = scalerX.transform(x_val)
x_test = scalerX.transform(x_test)

y_train = scalerY.transform(y_train)
y_val = scalerY.transform(y_val)
y_test = scalerY.transform(y_test)

class RidgeRegressionHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        reg = hp.Float('regularizer', min_value=1e-6, max_value=1, sampling="log")
        model.add(keras.layers.Input(shape = (m_features,)))
        model.add(keras.layers.Dense(3, kernel_regularizer= keras.regularizers.l2(reg), activation= None))
        model.compile(loss='mean_squared_error', optimizer = keras.optimizers.SGD(hp.Float("lr", min_value=1e-6, max_value=1e-1, sampling="log")))
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int("batch_size", min_value = 1, max_value = 25),
            **kwargs,
        )


tuner = keras_tuner.RandomSearch(
    RidgeRegressionHyperModel(),
    max_trials=500,
    # Do not resume the previous search in the same directory.
    overwrite=True,
    objective='val_loss',
    # Set a directory to store the intermediate results.
    directory="/tmp/tb",
)

tuner.search(x_train, y_train, epochs=100, validation_data=(x_val, y_val))
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best L2 lambda: {best_hps.get('regularizer')}")
print(f"Best Learning Rate: {best_hps.get('lr')}")
print(f"Best Batch Size: {best_hps.get('batch_size')}")

best_model = tuner.hypermodel.build(best_hps)
log_train = best_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=best_hps.get('batch_size')
)

plt.plot(log_train.history['loss'])
plt.plot(log_train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()