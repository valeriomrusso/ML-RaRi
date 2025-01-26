from load_data import *
from plots import plot_training_history_CUP
from model_builder import build_model_ridge_fixed
from model_training import train_model_fixed
from build_csv import csv_builder

def main():
    task = 'CUP'

    if task == 'CUP':
        filepath = './datasets/ML-CUP24-TR.csv'
        X_train, X_test, Y_train, Y_test, _, _ = load_and_preprocess_data_CUP(filepath)
    elif task == 'MONK':
        for monk in ['monks-1', 'monks-2', 'monks-3']:
            filepath = monk
            X_train, X_test, Y_train, Y_test = splitted_monk_data(filepath)

    reg = 0.00005
    learning_rate = 0.002
    momentum = 0.9
    batch_size = 80

    final_model = build_model_ridge_fixed(reg, learning_rate, momentum, task)
    history = train_model_fixed(final_model, batch_size, X_train, X_test, Y_train, Y_test)
    plot_training_history_CUP(history)

    final_dict = {}
    final_dict['batch_size'] = batch_size
    final_dict['reg'] = reg
    final_dict['learning_rate'] = learning_rate
    final_dict['momentum'] = momentum
    final_dict['train_loss'] = history.history['loss'][-1]
    final_dict['test_loss'] = history.history['val_loss'][-1]
    csv_builder(final_dict, 'best_hps_model_fixed.csv')

if __name__ == "__main__":
    main()
