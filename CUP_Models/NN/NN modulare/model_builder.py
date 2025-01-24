from tensorflow import keras

def build_model(hp, batch_size, units, dropout, num_layers, units_hidden, learning_rate, momentum):
    """Costruisce il modello con iperparametri configurabili."""
    model = keras.Sequential()
    
    hp.Int('batch_size', min_value=16, max_value=128, step=16)

    # Livello di input esplicito
    model.add(keras.layers.Input(shape=(12,)))
    
    # Primo livello denso
    model.add(keras.layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu'
    ))
    
    # Dropout
    model.add(keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1)))
    
    num_layers = hp.Int('num_layers', min_value=1, max_value=10, step=1)  # Ricerca da 1 a 10 layer
    for _ in range(num_layers):
        model.add(keras.layers.Dense(
            units=hp.Int('units_hidden', min_value=32, max_value=256, step=32),
            activation='relu'
        ))
        model.add(keras.layers.Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))

    # Livello di output
    model.add(keras.layers.Dense(3))
    
    # Compilazione
    model.compile(
        optimizer=keras.optimizers.SGD(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'),
            momentum=hp.Float('momentum', 0.0, 0.99, step=0.1)
        ),
        loss='mse'
    )
    
    return model
