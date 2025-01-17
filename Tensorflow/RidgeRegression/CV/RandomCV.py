import keras
import numpy as np
import keras_tuner
from sklearn.preprocessing import StandardScaler
import sklearn
import shutil
import os
from sklearn.model_selection import KFold
shutil.rmtree("/tmp/tb_logs", ignore_errors=True)
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
   
def build_model(hp):
  model = keras.Sequential()
  reg = hp.Float('regularizer', min_value=1e-6, max_value=1e-1, sampling="log")
  model.add(keras.layers.Input(shape = (m_features,)))
  model.add(keras.layers.Dense(3, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01), bias_initializer=keras.initializers.Zeros(), kernel_regularizer= keras.regularizers.l2(reg), activation= None))
  model.compile(loss='mean_squared_error', optimizer = keras.optimizers.SGD(hp.Float("lr", min_value=1e-6, max_value=1, sampling="log")))
  return model

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Risultati di tutte le fold
all_scores = []
top_hyperparam = {}

# K-Fold CV con Keras Tuner
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = Y[train_idx], Y[val_idx]
    
    scalerX = StandardScaler().fit(X_train)
    scalerY = StandardScaler().fit(y_train)

    X_train = scalerX.transform(X_train)
    X_val = scalerX.transform(X_val)

    y_train = scalerY.transform(y_train)
    y_val = scalerY.transform(y_val)
    # Configurazione di Keras Tuner
    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=20,
        directory='/tmp/tb_logs',
        project_name=f'fold_{fold+1}'
    )
    
    # Esecuzione del tuning
    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=10, verbose=1)
    
    # Miglior modello
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    
    # Addestramento e valutazione finale sulla fold
    best_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1, validation_data=(X_val, y_val))
    score = best_model.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold + 1} val_loss: {score}")
    all_scores.append(score)
    top_hyperparam[score] = best_model
# Risultati finali
print("Mean accuracy across folds:", np.mean(all_scores))
