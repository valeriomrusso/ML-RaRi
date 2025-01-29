from sklearn.model_selection import KFold
from model_builder import *
from model_training import *
from build_csv import csv_builder, original_scale
import keras
import pandas as pd

#Performs k-fold cross-validation to find the best hyperparameters for the given model.
#Trains a final model using the best hyperparameters found during cross-validation.
def CV(X_train, X_test, Y_train, Y_test, task, model, path, scalerY):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    hyperparameters_summary = []
    best_val_loss = float('inf')  
    for train_idx, val_idx in kfold.split(X_train):
        # Split the training data into training and validation sets for this fold
        x_train, x_val = X_train[train_idx], X_train[val_idx]
        y_train, y_val = Y_train[train_idx], Y_train[val_idx]

        # Train the model based on the specified type
        if model == "Ridge":
            trmodel, tuner = train_model_ranged(fold_no, build_model_ridge_ranged_tuner, x_train, y_train, x_val, y_val, task)
        elif model == "NN":
            trmodel, tuner = train_model_ranged(fold_no, build_model_nn_ranged_tuner, x_train, y_train, x_val, y_val, task)
        
        # Get the best hyperparameters and the best model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        batch_size = best_hps.get('batch_size')
        
        best_model = tuner.get_best_models(num_models=1)[0]
        # Evaluate the best model on the validation and training sets
        val_result= best_model.evaluate(x_val, y_val, verbose=0)
        train_result = best_model.evaluate(x_train, y_train, verbose=0)
        
        best_hps_dict = best_hps.values
        best_hps_dict['fold'] = fold_no  
        
        if task == 'MONK':
            best_hps_dict['accuracy'] =  train_result[1]
            best_hps_dict['val_accuracy'] = val_result[1]
            best_hps_dict['mse'] = train_result[2]
            best_hps_dict['val_mse'] = val_result[2]
            # Track the best validation accuracy
            if val_result[1] < best_val_loss:
                best_val_loss = val_result[1]
                best_hparams = best_hps
        elif task == 'CUP':
            best_hps_dict['mse'] = original_scale(train_result[1], scalerY)
            best_hps_dict['val_mse'] = original_scale(val_result[1], scalerY)
            best_hps_dict['mean_euclidean_error'] = original_scale(train_result[2], scalerY)
            best_hps_dict['val_mean_euclidean_error'] = original_scale(val_result[2], scalerY)
            # Track the best validation mean euclidean error
            if val_result[2] < best_val_loss:
                best_val_loss = val_result[2]
                best_hparams = best_hps
        hyperparameters_summary.append(best_hps_dict)
    
        fold_no += 1
    # Save the hyperparameters summary of all folds to a CSV file
    csv_builder(f'{path}/best_hyperparameters_per_fold.csv', hyperparameters_summary)

    # Build the final model using the best hyperparameters found during cross-validation
    if model == "Ridge":
        build_model = build_model_ridge_ranged_tuner(task)
        final_model = build_model(best_hparams)
    elif model == "NN":
        build_model = build_model_nn_ranged_tuner(task)
        final_model = build_model(best_hparams)
    
    # Train the final model on the entire training set
    final_history = final_model.fit(
        X_train, Y_train,
        batch_size=best_hparams.get('batch_size'),
        epochs=2000,
        validation_data=(X_test, Y_test),
        callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)]
    )
    final_dict = best_hparams.values
    if task == 'CUP':
        final_dict['mse'] = original_scale(final_history.history['mse'][-1], scalerY)
        final_dict['val_mse'] = original_scale(final_history.history['val_mse'][-1], scalerY)
        final_dict['mean_euclidean_error'] = original_scale(final_history.history['mean_euclidean_error'][-1], scalerY)
        final_dict['val_mean_euclidean_error'] = original_scale(final_history.history['val_mean_euclidean_error'][-1], scalerY)
    elif task == 'MONK':
        final_dict['accuracy'] = final_history.history['accuracy'][-1]
        final_dict['val_accuracy'] = final_history.history['val_accuracy'][-1]  
        final_dict['mse'] = final_history.history['mse'][-1]
        final_dict['val_mse'] = final_history.history['val_mse'][-1]
    # Save the final model's hyperparameters and metrics to a CSV file
    csv_builder(f'{path}/best_hyperparameters_final_model.csv', final_dict)

    # Return the training history and the final model
    return final_history, final_model