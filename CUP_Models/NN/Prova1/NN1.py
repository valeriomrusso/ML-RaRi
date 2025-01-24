# Imports
import keras
import numpy as np
import keras_tuner as kt
from keras import activations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Data loading and preprocessing
my_data = np.genfromtxt('ML-CUP24-TR.csv', delimiter=',')
X = my_data[:, 1:13]
Y = my_data[:, 13:16]
m_features = X.shape[1]

# Train/val/test split
train_ratio, validation_ratio, test_ratio = 0.60, 0.20, 0.20
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

# Scaling
scalerX, scalerY = StandardScaler(), StandardScaler()
x_train = scalerX.fit_transform(x_train)
x_val = scalerX.transform(x_val)
x_test = scalerX.transform(x_test)
y_train = scalerY.fit_transform(y_train)
y_val = scalerY.transform(y_val)
y_test = scalerY.transform(y_test)

# Model definition
def build_model(hp):
    model = keras.Sequential()
    reg = hp.Float('regularizer', min_value=1e-6, max_value=1e-1, sampling="log")
    model.add(keras.layers.Input(shape=(m_features,)))
    for units in [64, 128, 128, 64]:
        model.add(keras.layers.Dense(units,
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                   bias_initializer=keras.initializers.Zeros(),
                                   kernel_regularizer=keras.regularizers.l2(reg),
                                   activation=activations.relu))
    model.add(keras.layers.Dense(3,
                               kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                               bias_initializer=keras.initializers.Zeros(),
                               kernel_regularizer=keras.regularizers.l2(reg)))
    model.compile(loss='mean_squared_error',
                 optimizer=keras.optimizers.Adam(hp.Float("lr", min_value=1e-6, max_value=1, sampling="log")))
    return model

# Hyperparameter tuning
class MyTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', min_value=1, max_value=64)
        return super(MyTuner, self).run_trial(trial, *args, **kwargs)

tuner = MyTuner(build_model, max_trials=100, overwrite=True, objective='val_loss')
tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val),
            callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)])

# Final model training
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = build_model(best_hps)
x_train_full = np.concatenate([x_train, x_val])
y_train_full = np.concatenate([y_train, y_val])
history = model.fit(x_train_full, y_train_full,
                   batch_size=best_hps.get('batch_size'),
                   epochs=200,
                   validation_data=(x_test, y_test),
                   callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)])

# Plotting results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Test Loss')
ax1.set_title('Training and Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax1.legend()

window_size = 5
smoothed_train = pd.Series(history.history['loss']).rolling(window=window_size).mean()
smoothed_test = pd.Series(history.history['val_loss']).rolling(window=window_size).mean()
ax2.plot(smoothed_train, 'r-', label=f'Smoothed Train (window={window_size})')
ax2.plot(smoothed_test, 'b-', label=f'Smoothed Test (window={window_size})')
ax2.set_title('Smoothed Losses')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Save results
trials = tuner.oracle.get_best_trials(num_trials=100)
results = [{**t.hyperparameters.values, 
           'val_loss': t.score,
           'final_train_loss': history.history['loss'][-1] if t == trials[0] else None,
           'final_test_loss': history.history['val_loss'][-1] if t == trials[0] else None} 
          for t in trials]
pd.DataFrame(results).to_csv('results.csv', index=False)