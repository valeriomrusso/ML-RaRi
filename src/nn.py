from load_data import *
from plots import *
from cross_validation import CV
from model_builder import build_model_nn_fixed
from model_training import train_model_fixed
from build_csv import csv_builder
import os
from datetime import datetime

def NN(task, monktype=None, fixed=None, units=None, dropout=None, num_layers= None, units_hidden= None, learning_rate = None, momentum = None, reg = None, batch_size = None):

    if task == 'CUP':
        filepath = './datasets/ML-CUP24-TR.csv'
        X_train, X_test, Y_train, Y_test, _, _ = load_and_preprocess_data_CUP(filepath)
        name = task
    elif task == 'MONK':
        filepath = f"monks-{monktype}"
        X_train, X_test, Y_train, Y_test = splitted_monk_data(filepath)
        name = f'{task}{monktype}'

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if fixed:
        p = "fixed"
    else:
        p = "ranged"
    path = f'./results/nn_{p}_{name}_{timestamp}'
    os.makedirs(path, exist_ok=True)

    if fixed:
        final_model = build_model_nn_fixed(units, dropout, num_layers, units_hidden, learning_rate, momentum, reg, task)
        history = train_model_fixed(final_model, batch_size, X_train, X_test, Y_train, Y_test)
        final_dict = {}
        final_dict['batch_size'] = batch_size
        final_dict['units'] = units
        final_dict['dropout'] = dropout
        final_dict['num_layers'] = num_layers
        final_dict['units_hidden'] = units_hidden
        final_dict['learning_rate'] = learning_rate
        final_dict['momentum'] = momentum
        final_dict['reg'] = reg
        final_dict['train_loss'] = history.history['loss'][-1]
        final_dict['test_loss'] = history.history['val_loss'][-1]
        if task == 'MONK':
            final_dict['accuracy'] = history.history['accuracy'][-1]
            final_dict['val_accuracy'] = history.history['val_accuracy'][-1]  
            final_dict['mse'] = history.history['mse'][-1]
            final_dict['val_mse'] = history.history['val_mse'][-1]
        csv_builder(f'{path}/best_hps_model_fixed.csv', final_dict)
    else:
        history, model = CV(X_train, X_test, Y_train, Y_test, task, "NN", path)
    
    if task == 'CUP':
        plot_training_history_CUP(history, path)
    elif task == 'MONK':
        plot_training_history_Monk(history, path)
