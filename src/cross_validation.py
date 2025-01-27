from sklearn.model_selection import KFold
from model_builder import *
from model_training import *
from build_csv import csv_builder
import keras
import pandas as pd

def CV(X_train, X_test, Y_train, Y_test, task, model, path):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    hyperparameters_summary = []
    best_val_loss = float('inf')  # Inizializza con infinito
    for train_idx, val_idx in kfold.split(X_train):
        x_train, x_val = X_train[train_idx], X_train[val_idx]
        y_train, y_val = Y_train[train_idx], Y_train[val_idx]

        if model == "Ridge":
            trmodel, tuner = train_model_ranged(fold_no, build_model_ridge_ranged_tuner, x_train, y_train, x_val, y_val, task)
        elif model == "NN":
            trmodel, tuner = train_model_ranged(fold_no, build_model_nn_ranged_tuner, x_train, y_train, x_val, y_val, task)
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        batch_size = best_hps.get('batch_size')
        
        best_model = tuner.get_best_models(num_models=1)[0]
        
        val_result= best_model.evaluate(x_val, y_val, verbose=0)
        train_result = best_model.evaluate(x_train, y_train, verbose=0)
        
        best_hps_dict = best_hps.values
        best_hps_dict['fold'] = fold_no  # Aggiungi il numero del fold
        
        if task == 'MONK':
            best_hps_dict['accuracy'] =  train_result[1]
            best_hps_dict['val_accuracy'] = val_result[1]
            best_hps_dict['mse'] = train_result[2]
            best_hps_dict['val_mse'] = val_result[2]
            if best_hps_dict['val_accuracy'] < best_val_loss:
                best_val_loss = best_hps_dict['val_accuracy']
                best_hparams = best_hps
        elif task == 'CUP':
            best_hps_dict['mse'] = train_result[1]
            best_hps_dict['val_mse'] = val_result[1]
            if best_hps_dict['val_mse'] < best_val_loss:
                best_val_loss = best_hps_dict['val_mse']
                best_hparams = best_hps
        hyperparameters_summary.append(best_hps_dict)
    
        fold_no += 1

    csv_builder(f'{path}/best_hyperparameters_per_fold.csv', hyperparameters_summary)

    if model == "Ridge":
        build_model = build_model_ridge_ranged_tuner(task)
        final_model = build_model(best_hparams)
    elif model == "NN":
        build_model = build_model_nn_ranged_tuner(task)
        final_model = build_model(best_hparams)
    
    final_history = final_model.fit(
        X_train, Y_train,
        batch_size=best_hparams.get('batch_size'),
        epochs=200,  # Puoi regolare il numero di epoche
        validation_data=(X_test, Y_test),  # Usa il test set per la validazione finale
        callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)]
    )
    final_dict = best_hparams.values
    final_dict['mse'] = final_history.history['mse'][-1]
    final_dict['val_mse'] = final_history.history['val_mse'][-1]
    if task == 'MONK':
        final_dict['accuracy'] = final_history.history['accuracy'][-1]
        final_dict['val_accuracy'] = final_history.history['val_accuracy'][-1]  
    csv_builder(f'{path}/best_hyperparameters_final_model.csv', final_dict)
    return final_history, final_model