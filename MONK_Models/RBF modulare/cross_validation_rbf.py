from sklearn.model_selection import KFold
from build_rbf_tuning import build_rbf_model_tuner_CUP, build_rbf_model_tuner_MONK
from model_training import train_model
import keras
import pandas as pd

def CV(X_train, X_test, Y_train, Y_test, task, model):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    hyperparameters_summary = []
    best_val_loss = float('inf')  # Inizializza con infinito
    best_history = None
    for train_idx, val_idx in kfold.split(X_train):
        x_train, x_val = X_train[train_idx], X_train[val_idx]
        y_train, y_val = Y_train[train_idx], Y_train[val_idx]
        
        if task == "CUP" and model == "RBF":
            model, tuner = train_model(fold_no, build_rbf_model_tuner_CUP, x_train, y_train, x_val, y_val)
        elif task == "MONK" and model == "RBF":
            model, tuner = train_model(fold_no, build_rbf_model_tuner_MONK, x_train, y_train, x_val, y_val)
        elif task == "CUP" and model == "NN":
            model, tuner = train_model(fold_no, build_rbf_model_tuner_CUP, x_train, y_train, x_val, y_val)
        elif task == "MONK" and model == "NN":
            model, tuner = train_model(fold_no, build_rbf_model_tuner_CUP, x_train, y_train, x_val, y_val)

        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        batch_size = best_hps.get('batch_size')
        
        train_loss = model.evaluate(x_train, y_train, verbose=0)
        val_loss = model.evaluate(x_val, y_val, verbose=0)
        
        best_hps_dict = best_hps.values
        best_hps_dict['fold'] = fold_no  # Aggiungi il numero del fold
        best_hps_dict['train_loss'] = train_loss
        best_hps_dict['val_loss'] = val_loss

        hyperparameters_summary.append(best_hps_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hparams = best_hps
    
        fold_no += 1
    
    hyperparameters_df = pd.DataFrame(hyperparameters_summary)
    hyperparameters_df.to_csv('best_hyperparameters_per_fold.csv', index=False)
    final_model = build_rbf_model_tuner_CUP(best_hparams)
    final_history = final_model.fit(
        X_train, Y_train,
        batch_size=best_hps.get('batch_size'),
        epochs=200,  # Puoi regolare il numero di epoche
        validation_data=(X_test, Y_test),  # Usa il test set per la validazione finale
        callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)]
    )

    final_dict = best_hparams.values
    final_dict['train_loss'] = final_history.history['loss'][-1]
    final_dict['val_loss'] = final_history.history['val_loss'][-1]
    final_model_df = pd.DataFrame(final_dict, index=[0])
    final_model_df.to_csv('best_hyperparameters_final_model.csv', index=False)

    return final_history, final_model