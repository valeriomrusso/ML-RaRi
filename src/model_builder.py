from tensorflow import keras
from keras import regularizers

def build_model_nn_ranged_tuner(task):
    def build_model(hp):
        if task == 'CUP':
            input_shape = (12,)
            output_shape = 3
            actfun = 'linear'
            loss = 'mse'
            metrics=['mse']
        elif task == 'MONK':
            input_shape = (6,)
            output_shape = 1
            actfun = 'sigmoid'
            loss = 'binary_crossentropy'
            metrics=['accuracy', 'mse']
        
        """Costruisce il modello con iperparametri configurabili."""
        model = keras.Sequential()
        reg = hp.Float('lambda', 1e-6, 1e-2, sampling='log')

        batch_size = hp.Int('batch_size', 16, 128, step = 4)

        # Livello di input esplicito
        model.add(keras.layers.Input(shape=input_shape))
        
        # Primo livello denso
        model.add(keras.layers.Dense(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            activation='relu',
            kernel_regularizer = regularizers.l2(reg)
        ))
        
        # Dropout
        model.add(keras.layers.Dropout(hp.Float('dropout', 0, 0.3, step=0.1)))
        
        num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)  # Ricerca da 1 a 10 layer
        for _ in range(num_layers):
            model.add(keras.layers.Dense(
                units=hp.Int('units_hidden', min_value=32, max_value=128, step=32),
                activation='relu',
                kernel_regularizer = regularizers.l2(reg)
            ))
            model.add(keras.layers.Dropout(hp.Float('dropout', 0.0, 0.3, step=0.1)))

        # Livello di output
        model.add(keras.layers.Dense(output_shape, activation=actfun, kernel_regularizer = regularizers.l2(reg)))
        
        # Compilazione
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
            ),
            loss=loss,
            metrics=metrics
        )
        
        return model
    return build_model


def build_model_nn_fixed(units, dropout, num_layers, units_hidden, learning_rate, reg, task):
    """Costruisce il modello con iperparametri configurabili."""
    model = keras.Sequential()
    if task == 'CUP':
        input_shape = (12,)
        output_shape = 3
        actfun = 'linear'
        metrics=['mse']
        loss = 'mse'
    elif task == 'MONK':
        input_shape = (6,)
        output_shape = 1
        actfun = 'sigmoid'
        metrics=['accuracy', 'mse']
        loss = 'binary_crossentropy'
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
        optimizer=keras.optimizers.Adam(learning_rate), loss=loss, metrics=metrics)
    
    return model

def build_model_ridge_ranged_tuner(task):
    def build_model(hp):
        if task == 'CUP':
            input_shape = (12,)
            output_shape = 3
            actfun = 'linear'
            metrics=['mse']
            loss = 'mse'
        elif task == 'MONK':
            input_shape = (6,)
            output_shape = 1
            actfun = 'sigmoid'
            metrics=['accuracy', 'mse']
            loss = 'binary_crossentropy'

        
        batch_size = hp.Int('batch_size', 16, 128, step = 4)

        model = keras.Sequential()
        reg = hp.Float('regularizer', min_value=1e-6, max_value=1, sampling="log")
        model.add(keras.layers.Input(shape = input_shape))
        model.add(keras.layers.Dense(output_shape, kernel_regularizer= keras.regularizers.l2(reg), activation= actfun))
        model.compile(loss=loss, optimizer = keras.optimizers.Adam(hp.Float("lr", min_value=1e-6, max_value=1e-1, sampling="log")), metrics=metrics)
        return model
    return build_model

def build_model_ridge_fixed(reg, learning_rate, task):
    if task == 'CUP':
        input_shape = (12,)
        output_shape = 3
        actfun = 'linear'
        metrics=['mse']
        loss = 'mse'
    elif task == 'MONK':
        input_shape = (6,)
        output_shape = 1
        actfun = 'sigmoid'
        metrics=['accuracy', 'mse']
        loss = 'binary_crossentropy'

    model = keras.Sequential()
    model.add(keras.layers.Input(input_shape))
    model.add(keras.layers.Dense(output_shape, kernel_regularizer= keras.regularizers.l2(reg), activation= actfun))
    model.compile(loss=loss, optimizer = keras.optimizers.Adam(learning_rate), metrics=metrics)
    return model