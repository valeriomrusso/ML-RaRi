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
def build_rbf_model_tuner(hp):
    input_dim = 6  # Number of input features
    output_dim = 1  # Number of targets (x, y, z)

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
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mse'])

    return model


def rbf_tuning(X_train, y_train, X_val, y_val, X_test, y_test):
    # Hyperparameter tuning
    results = []  # To store hyperparameters and results

    def record_results(hyperparameters, test_mse, accuracy):
        entry = hyperparameters.copy()
        entry['test_mse'] = test_mse
        entry['accuracy'] = accuracy
        results.append(entry)

    # Use a temporary directory
    temp_dir = tempfile.mkdtemp()

    tuner = kt.Hyperband(
        build_rbf_model_tuner,
        objective='accuracy',
        max_epochs=100,
        factor=3,
        overwrite=True,
        directory=temp_dir
    )

    # Search for the best hyperparameters
    tuner.search(
        X_train, y_train,
        epochs=100,
        validation_data=(X_val, y_val)
    )

    # Get all tested hyperparameters
    all_hps = tuner.get_best_hyperparameters(num_trials=10)  # Modify as needed
    for hp in all_hps:
        best_model = tuner.hypermodel.build(hp)
        history = best_model.fit(
            X_train, y_train,
            epochs=100,  # Use a small number to quickly evaluate
            batch_size=hp.get('batch_size'),
            validation_data=(X_val, y_val),
            verbose=0
        )

        # Evaluate on test set
        test_mse = min(history.history['mse'])
        accuracy = max(history.history['accuracy'])

        # Record the results
        record_results(hp.values, test_mse, accuracy)

    # Plot the results for the best configuration
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    history = best_model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=best_hp['batch_size'],
        validation_data=(X_val, y_val)
    )

    return history, best_hp.values