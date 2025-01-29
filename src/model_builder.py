import keras
from keras import regularizers
import tensorflow as tf
import numpy as np

# Function to build a neural network model with tunable hyperparameters
def build_model_nn_ranged_tuner(task):
    def build_model(hp):
        if task == 'CUP':
            input_shape = (12,)
            output_shape = 3
            actfun = 'linear'
            loss = 'mse'
            metrics=['mse', mean_euclidean_error]
            units = hp.Int('units', min_value=32, max_value=128, step=32)
            num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
        elif task == 'MONK':
            input_shape = (17,)
            output_shape = 1
            actfun = 'sigmoid'
            loss = 'binary_crossentropy'
            metrics=['accuracy', 'mse']
            units = 4
            num_layers = 0
        
        model = keras.Sequential()
        reg = hp.Float('lambda', 1e-6, 1e-2, sampling='log')

        model.add(keras.layers.Input(shape=input_shape))
        
        model.add(keras.layers.Dense(
            units=units,
            activation='relu',
            kernel_regularizer = regularizers.l2(reg)
        ))
        
        model.add(keras.layers.Dropout(hp.Float('dropout', 0, 0.3, step=0.1)))
        
        for _ in range(num_layers):
            model.add(keras.layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer = regularizers.l2(reg)
            ))
            model.add(keras.layers.Dropout(hp.Float('dropout', 0.0, 0.3, step=0.1)))

        model.add(keras.layers.Dense(output_shape, activation=actfun, kernel_regularizer = regularizers.l2(reg)))
        
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
            ),
            loss=loss,
            metrics=metrics
        )
        
        return model
    return build_model

# Function to build a neural network with fixed hyperparameters
def build_model_nn_fixed(units, dropout, num_layers, learning_rate, reg, task):
    model = keras.Sequential()
    if task == 'CUP':
        input_shape = (12,)
        output_shape = 3
        actfun = 'linear'
        metrics=['mse', mean_euclidean_error]
        loss = 'mse'
    elif task == 'MONK':
        input_shape = (17,)
        output_shape = 1
        actfun = 'sigmoid'
        metrics=['accuracy', 'mse']
        loss = 'binary_crossentropy'
    
    model.add(keras.layers.Input(input_shape))
    
    model.add(keras.layers.Dense(
        units=units,
        activation='relu',
        kernel_regularizer = regularizers.l2(reg)
    ))
    
    model.add(keras.layers.Dropout(dropout))
    
    for _ in range(num_layers):
        model.add(keras.layers.Dense(
            units=units,
            activation='relu',
            kernel_regularizer = regularizers.l2(reg)
        ))
        model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(output_shape, activation=actfun, kernel_regularizer = regularizers.l2(reg)))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate), loss=loss, metrics=metrics)
    
    return model

# Function to build a ridge regression model with tunable hyperparameters
def build_model_ridge_ranged_tuner(task):
    def build_model(hp):
        if task == 'CUP':
            input_shape = (12,)
            output_shape = 3
            actfun = 'linear'
            metrics=['mse', mean_euclidean_error]
            loss = 'mse'
        elif task == 'MONK':
            input_shape = (17,)
            output_shape = 1
            actfun = 'sigmoid'
            metrics=['accuracy', 'mse']
            loss = 'binary_crossentropy'

        model = keras.Sequential()
        reg = hp.Float('regularizer', min_value=1e-6, max_value=1, sampling="log")
        model.add(keras.layers.Input(shape = input_shape))
        model.add(keras.layers.Dense(output_shape, kernel_regularizer= keras.regularizers.l2(reg), activation= actfun))
        model.compile(loss=loss, optimizer = keras.optimizers.Adam(hp.Float("lr", min_value=1e-6, max_value=1e-1, sampling="log")), metrics=metrics)
        return model
    return build_model

# Function to build a ridge regression model with fixed hyperparameters
def build_model_ridge_fixed(reg, learning_rate, task):
    if task == 'CUP':
        input_shape = (12,)
        output_shape = 3
        actfun = 'linear'
        metrics=['mse', mean_euclidean_error]
        loss = 'mse'
    elif task == 'MONK':
        input_shape = (17,)
        output_shape = 1
        actfun = 'sigmoid'
        metrics=['accuracy', 'mse']
        loss = 'binary_crossentropy'

    model = keras.Sequential()
    model.add(keras.layers.Input(input_shape))
    model.add(keras.layers.Dense(output_shape, kernel_regularizer= keras.regularizers.l2(reg), activation= actfun))
    model.compile(loss=loss, optimizer = keras.optimizers.Adam(learning_rate), metrics=metrics)
    return model

# Custom metric: Mean Euclidean Error
def mean_euclidean_error(y_true, y_pred):
    diff = tf.square(y_pred - y_true)
    mean_diff = tf.reduce_mean(diff, axis=-1)
    return tf.sqrt(mean_diff)
