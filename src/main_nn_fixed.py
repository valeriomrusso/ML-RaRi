from load_data import *
from plots import plot_training_history_CUP
from model_builder import build_model_nn_fixed
from model_training import train_model_fixed
from build_csv import csv_builder
import os
from datetime import datetime

def main():
    task = 'CUP'
    monktype = 1

    units = 160
    dropout = 0.1
    num_layers = 4
    units_hidden = 224
    learning_rate = 0.002
    momentum = 0.9
    reg = 0.00005
    batch_size = 80
    
    if task == 'CUP':
        filepath = './datasets/ML-CUP24-TR.csv'
        X_train, X_test, Y_train, Y_test, _, _ = load_and_preprocess_data_CUP(filepath)
    elif task == 'MONK':
        filepath = f"monks-{monktype}"
        X_train, X_test, Y_train, Y_test = splitted_monk_data(filepath)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f'./results/nn_fixed_{task}_{timestamp}'
    os.makedirs(path, exist_ok=True)

    final_model = build_model_nn_fixed(units, dropout, num_layers, units_hidden, learning_rate, momentum, reg, task)
    history = train_model_fixed(final_model, batch_size, X_train, X_test, Y_train, Y_test)
    plot_training_history_CUP(history, path)

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
    csv_builder(f'{path}/best_hps_model_fixed.csv', final_dict)


if __name__ == "__main__":
    main()
