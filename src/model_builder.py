from tensorflow import keras
from keras import regularizers

def build_model_nn_ranged(hp, task):
    
    if task == 'CUP':
        input_shape = (12,)
        output_shape = 3
        actfun = 'linear'
    elif task == 'MONK':
        input_shape = (6,)
        output_shape = 1
        actfun = 'sigmoid'
    
    """Costruisce il modello con iperparametri configurabili."""
    model = keras.Sequential()
    reg = hp.Float('lambda', 1e-6, 1, sampling='log')
    hp.Int('batch_size', min_value=16, max_value=128, step=16)

    # Livello di input esplicito
    model.add(keras.layers.Input(shape=input_shape))
    
    # Primo livello denso
    model.add(keras.layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu',
        kernel_regularizer = regularizers.l2(reg)
    ))
    
    # Dropout
    model.add(keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1)))
    
    num_layers = hp.Int('num_layers', min_value=1, max_value=10, step=1)  # Ricerca da 1 a 10 layer
    for _ in range(num_layers):
        model.add(keras.layers.Dense(
            units=hp.Int('units_hidden', min_value=32, max_value=256, step=32),
            activation='relu',
            kernel_regularizer = regularizers.l2(reg)
        ))
        model.add(keras.layers.Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))

    # Livello di output
    model.add(keras.layers.Dense(output_shape, activation=actfun, kernel_regularizer = regularizers.l2(reg)))
    
    # Compilazione
    model.compile(
        optimizer=keras.optimizers.SGD(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'),
            momentum=hp.Float('momentum', 0.0, 0.99, step=0.1)
        ),
        loss='mse'
    )
    
    return model


def build_model_nn_fixed(units, dropout, num_layers, units_hidden, learning_rate, momentum, reg, task):
    """Costruisce il modello con iperparametri configurabili."""
    model = keras.Sequential()
    if task == 'CUP':
        input_shape = (12,)
        output_shape = 3
        actfun = 'linear'
    elif task == 'MONK':
        input_shape = (6,)
        output_shape = 1
        actfun = 'sigmoid'
    # Livello di input esplicito
    model.add(keras.layers.Input(input_shape))
    
    # Primo livello denso
    model.add(keras.layers.Dense(
        units=units,
        activation='relu',
        kernel_regularizer = regularizers.l2(reg)
    ))
    
    # Dropout
    model.add(keras.layers.Dropout(dropout))
    
    for _ in range(num_layers):
        model.add(keras.layers.Dense(
            units=units_hidden,
            activation='relu',
            kernel_regularizer = regularizers.l2(reg)
        ))
        model.add(keras.layers.Dropout(dropout))

    # Livello di output
    model.add(keras.layers.Dense(output_shape, activation=actfun, kernel_regularizer = regularizers.l2(reg)))
    
    # Compilazione
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate, momentum), loss='mse')
    
    return model

def build_model_ridge_ranged(hp, task):
    if task == 'CUP':
        input_shape = (12,)
        output_shape = 3
        actfun = 'linear'
    elif task == 'MONK':
        input_shape = (6,)
        output_shape = 1
        actfun = 'sigmoid'

    model = keras.Sequential()
    batch_size=hp.Int('batch_size', min_value = 1, max_value = 25)
    reg = hp.Float('regularizer', min_value=1e-6, max_value=1, sampling="log")
    model.add(keras.layers.Input(shape = input_shape))
    model.add(keras.layers.Dense(output_shape, kernel_regularizer= keras.regularizers.l2(reg), activation= actfun))
    model.compile(loss='mse', optimizer = keras.optimizers.SGD(hp.Float("lr", min_value=1e-6, max_value=1e-1, sampling="log")))
    return model

def build_model_ridge_fixed(reg, learning_rate, momentum, task):
    if task == 'CUP':
        input_shape = (12,)
        output_shape = 3
        actfun = 'linear'
    elif task == 'MONK':
        input_shape = (6,)
        output_shape = 1
        actfun = 'sigmoid'

    model = keras.Sequential()
    model.add(keras.layers.Input(input_shape))
    model.add(keras.layers.Dense(output_shape, kernel_regularizer= keras.regularizers.l2(reg), activation= actfun))
    model.compile(loss='mse', optimizer = keras.optimizers.SGD(learning_rate, momentum))
    return model