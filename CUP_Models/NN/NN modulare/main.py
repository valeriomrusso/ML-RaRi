from sklearn.model_selection import KFold, train_test_split
from load_CUP import load_and_preprocess_data
from model_builder import build_model
from model_training import train_model
from plots import plot_training_history
import keras
import pandas as pd

def main():
    filepath = 'ML-CUP24-TR.csv'
    X, Y, _, _ = load_and_preprocess_data(filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    hyperparameters_summary = []
    best_val_loss = float('inf')  # Inizializza con infinito
    best_history = None
    for train_idx, val_idx in kfold.split(X_train):
        x_train, x_val = X_train[train_idx], X_train[val_idx]
        y_train, y_val = Y_train[train_idx], Y_train[val_idx]
        
        model, tuner = train_model(fold_no, build_model, x_train, y_train, x_val, y_val)
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        batch_size = best_hps.get('batch_size')
        
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=200,
            validation_data=(x_val, y_val),
            callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)]
        )
        
        best_hps_dict = best_hps.values
        best_hps_dict['fold'] = fold_no  # Aggiungi il numero del fold
        best_hps_dict['train_loss'] = history.history['loss'][-1]
        best_hps_dict['val_loss'] = history.history['val_loss'][-1]
        hyperparameters_summary.append(best_hps_dict)

        if history.history['val_loss'][-1]< best_val_loss:
            best_val_loss = history.history['val_loss'][-1]
            best_hparams = best_hps
    
        fold_no += 1
    
    hyperparameters_df = pd.DataFrame(hyperparameters_summary)
    hyperparameters_df.to_csv('best_hyperparameters_per_fold.csv', index=False)
    final_model = build_model(best_hparams)
    final_history = final_model.fit(
        X_train, Y_train,
        batch_size=best_hps.get('batch_size'),
        epochs=200,  # Puoi regolare il numero di epoche
        validation_data=(X_test, Y_test),  # Usa il test set per la validazione finale
        callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)]
    )

    plot_training_history(final_history)


if __name__ == "__main__":
    main()
