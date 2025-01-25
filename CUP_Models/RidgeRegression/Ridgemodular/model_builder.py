from tensorflow import keras

def build_model(hp):
  model = keras.Sequential()
  batch_size=hp.Int('batch_size', min_value = 1, max_value = 25)
  reg = hp.Float('regularizer', min_value=1e-6, max_value=1, sampling="log")
  model.add(keras.layers.Input(shape = (12,)))
  model.add(keras.layers.Dense(3, kernel_regularizer= keras.regularizers.l2(reg), activation= None))
  model.compile(loss='mse', optimizer = keras.optimizers.SGD(hp.Float("lr", min_value=1e-6, max_value=1e-1, sampling="log")))
  return model
dsdf

def build_model_fixed(reg, learning_rate, momentum):
  model = keras.Sequential()
  model.add(keras.layers.Input(shape = (12,)))
  model.add(keras.layers.Dense(3, kernel_regularizer= keras.regularizers.l2(reg), activation= None))
  model.compile(loss='mse', optimizer = keras.optimizers.SGD(learning_rate, momentum))
  return model