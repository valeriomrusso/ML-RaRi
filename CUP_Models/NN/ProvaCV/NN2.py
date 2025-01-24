import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import keras_tuner as kt
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

m_features = 12

# Loading data
my_data = np.genfromtxt('ML-CUP24-TR.csv', delimiter=',')
X = my_data[:, 1:13]
Y = my_data[:, 13:16]
print(X.shape)
print(Y.shape)

scalerX = StandardScaler().fit(X)
scalerY = StandardScaler().fit(Y)

X = scalerX.transform(X)
Y = scalerY.transform(Y)

def build_model(hp):
    """Costruisce il modello con iperparametri configurabili"""
    model = keras.Sequential()
    
    # Parametri di training
    hp.Int('batch_size', min_value=32, max_value=128, step=32)
    
    # Layer
    model.add(keras.layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu',
        input_shape=(12,)
    ))
    model.add(keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1)))
    model.add(keras.layers.Dense(3))
    
    # Compilazione
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='mse'
    )
    return model

def train_model(x_train, y_train, x_val, y_val, x_test, y_test):
    """Training con gestione errori"""
    try:
        tuner = kt.Hyperband(
            build_model,
            objective='val_loss',
            max_epochs=200,
            directory='tuner_results',
            project_name='nn_tuning'
        )
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = build_model(best_hps)
        
        # Usa il batch_size dagli iperparametri
        batch_size = best_hps.get('batch_size')
        
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=200,
            validation_data=(x_val, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping('val_loss', patience=5)
            ]
        )
        
        return model, history, tuner
        
    except Exception as e:
        print(f"Errore durante il training: {str(e)}")
        raise

def plot_training_history(history, window_size=5):
    """
    Visualizza l'andamento del training
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot losses grezze
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Test Loss')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot losses smoothed
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

def save_results(tuner, history):
    """
    Salva i risultati del training
    """
    trials = tuner.oracle.get_best_trials(num_trials=100)
    results = [{
        **t.hyperparameters.values,
        'val_loss': t.score,
        'final_train_loss': history.history['loss'][-1] if t == trials[0] else None,
        'final_test_loss': history.history['val_loss'][-1] if t == trials[0] else None
    } for t in trials]
    
    pd.DataFrame(results).to_csv('results.csv', index=False)

def main():
    # Use KFold for cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    results = []

    for train_index, val_index in kfold.split(X):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]
        
        # Initialize and run the tuner
        model, history, tuner = train_model(x_train, y_train, x_val, y_val, x_val, y_val)
        
        # Get the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        trials = tuner.oracle.get_best_trials(num_trials=100)
        HP_list = [{**trial.hyperparameters.values, "Score": trial.score} for trial in trials]
        HP_df = pd.DataFrame(HP_list)
        HP_df.to_csv(f"fold_{fold_no}_hp_results.csv", index=False, na_rep='NaN')
        
        results.append(best_hps)
        
        # groove number increase
        fold_no += 1

    # Display best hyperparameters for each fold
    for i, best_hp in enumerate(results):
        print(f"Best hyperparameters for fold {i+1}: {best_hp.values}")

    # Final model training
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = build_model(best_hps)
    x_train_full = np.concatenate([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])
    history = model.fit(x_train_full, y_train_full,
                    batch_size=best_hps.get('batch_size'),
                    epochs=200,
                    validation_data=(x_val, y_val),
                    callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)])

    # Plotting results
    plot_training_history(history)

    # Save results
    save_results(tuner, history)

if __name__ == "__main__":
    main()