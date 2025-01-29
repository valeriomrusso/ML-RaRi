from load_data import *
from plots import *
from cross_validation import CV
from build_csv import csv_builder
import os
from model_builder import build_model_rbf_fixed
from model_training import train_model_fixed
from datetime import datetime

#Main function to train and evaluate the RBF network model based on task type ('CUP' or 'MONK').
def RBF(task, monktype=None, fixed=None, n_centers=None, gamma=None, learning_rate=None, batch_size=None):

    # Load and preprocess data based on the task type (CUP or MONK)
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
    path = f'./results/rbf_{p}_{name}_{timestamp}'
    os.makedirs(path, exist_ok=True)

    if fixed:
        # Build and train the model with the fixed hyperparameters
        final_model = build_model_rbf_fixed(n_centers, gamma, learning_rate, task)
        history = train_model_fixed(final_model, batch_size, X_train, X_test, Y_train, Y_test)
        final_dict = {}
        final_dict['batch_size'] = batch_size
        final_dict['n_centers'] = n_centers
        final_dict['gamma'] = gamma
        final_dict['learning_rate'] = learning_rate
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
        
        # Save the results to a CSV file
        csv_builder(f'{path}/best_hps_model_fixed.csv', final_dict)
    else:
        # Use cross-validation to tune the hyperparameters and train the model
        history, model = CV(X_train, X_test, Y_train, Y_test, task, "RBF", path, scalerY)
    
    # Plot the training history
    if task == 'CUP':
        plot_training_history_CUP(history, scalerY, path)
    elif task == 'MONK':
        plot_training_history_Monk(history, path)
