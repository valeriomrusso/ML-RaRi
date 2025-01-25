import tempfile
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import numpy as np

# Function to compute RBF activation (placeholder, implement your logic)
def rbf_activation(x, centers, gamma):
    # Example implementation for RBF activation
    return tf.exp(-gamma * tf.reduce_sum(tf.square(x[:, None, :] - centers), axis=-1))

# Function to build the model for Keras Tuner
def build_rbf_model_tuner_CUP(hp):
    input_dim = 12  # Number of input features
    output_dim = 3  # Number of targets (x, y, z)

    n_centers = hp.Int('n_centers', min_value=100, max_value=200, step=5)
    gamma = hp.Float('gamma', min_value=0.1, max_value=0.3, step=0.005)
    learning_rate = hp.Float('learning_rate', min_value=0.01, max_value=0.4, step=0.01)
    batch_size = hp.Int('batch_size', min_value=50, max_value=70, step=5)
    model = models.Sequential()

    # Layer di input: utilizza 'shape' invece di 'input_dim'
    model.add(layers.InputLayer(shape=(input_dim,)))

    # Layer nascosta con attivazione RBF (calcola distanza dal centro)
    centers = tf.Variable(np.random.randn(n_centers, input_dim), dtype=tf.float32)  # Centri randomici
    model.add(layers.Lambda(lambda x: rbf_activation(x, centers, gamma)))

    # Layer di output con una dimensione pari al numero di target (3 in questo caso: x, y, z)
    model.add(layers.Dense(output_dim))

    # Optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer, loss='mse')

    return model


# Function to build the model for Keras Tuner
def build_rbf_model_tuner_MONK(hp):
    input_dim = 6  # Number of input features
    output_dim = 1  # Number of targets (x, y, z)

    n_centers = hp.Int('n_centers', min_value=100, max_value=200, step=5)
    gamma = hp.Float('gamma', min_value=0.1, max_value=0.3, step=0.005)
    learning_rate = hp.Float('learning_rate', min_value=0.01, max_value=0.4, step=0.01)
    batch_size = hp.Int('batch_size', min_value=50, max_value=70, step=5)
    model = models.Sequential()

    # Layer di input: utilizza 'shape' invece di 'input_dim'
    model.add(layers.InputLayer(shape=(input_dim)))

    # Layer nascosta con attivazione RBF (calcola distanza dal centro)
    centers = tf.Variable(np.random.randn(n_centers, input_dim), dtype=tf.float32)  # Centri randomici
    model.add(layers.Lambda(lambda x: rbf_activation(x, centers, gamma)))

    # Layer di output con una dimensione pari al numero di target (3 in questo caso: x, y, z)
    model.add(layers.Dense(output_dim))

    # Optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer, loss='mse')

    return model