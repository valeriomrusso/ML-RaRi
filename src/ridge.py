from load_data import *
from plots import *
from model_builder import build_model_ridge_fixed
from model_training import train_model_fixed
from build_csv import *
from cross_validation import CV
import os
from datetime import datetime

def Ridge(task, monktype = None, fixed = None, learning_rate = None, momentum = None, reg = None, batch_size = None):
    if task == 'CUP':
        filepath = './datasets/ML-CUP24-TR.csv'
        X_train, X_test, Y_train, Y_test, scalerX, scalerY = load_and_preprocess_data_CUP(filepath)
        name = task
    elif task == 'MONK':
        filepath = f"monks-{monktype}"
        X_train, X_test, Y_train, Y_test = splitted_monk_data(filepath)
        scalerY = None 
        name = f'{task}{monktype}'

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if fixed:
        p = "fixed"
    else:
        p = "ranged"
    path = f'./results/ridge_{p}_{name}_{timestamp}'
    os.makedirs(path, exist_ok=True)

    if fixed:
        model = build_model_ridge_fixed(reg, learning_rate, momentum, task)
        history = train_model_fixed(model, batch_size, X_train, X_test, Y_train, Y_test)
        final_dict = {}
        final_dict['batch_size'] = batch_size
        final_dict['reg'] = reg
        final_dict['learning_rate'] = learning_rate
        final_dict['momentum'] = momentum
        if task == 'CUP':
            final_dict['mse'] = original_scale(history.history['mse'][-1], scalerY)
            final_dict['val_mse'] = original_scale(history.history['val_mse'][-1], scalerY)
            final_dict['mean_euclidean_error'] = original_scale(history.history['mean_euclidean_error'][-1], scalerY)
            final_dict['val_mean_euclidean_error'] = original_scale(history.history['val_mean_euclidean_error'][-1], scalerY)
        elif task == 'MONK':
            final_dict['accuracy'] = history.history['accuracy'][-1]
            final_dict['val_accuracy'] = history.history['val_accuracy'][-1]  
            final_dict['mse'] = history.history['mse'][-1]
            final_dict['val_mse'] = history.history['val_mse'][-1]
        csv_builder(f'{path}/best_hps_model_fixed.csv', final_dict)
    else:
        history, model = CV(X_train, X_test, Y_train, Y_test, task, "Ridge", path, scalerY)
    if task == 'CUP':
        plot_training_history_CUP(history, scalerY, path)
    elif task == 'MONK':
        plot_training_history_Monk(history, path)