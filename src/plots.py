import matplotlib.pyplot as plt
import numpy as np
from build_csv import original_scale
import pandas as pd
import datetime


def plot_training_history_CUP(history, scaler, path, window_size=5):
    """Visualizza l'andamento del training con riscalatura di MSE e MEE, riportando i valori alla scala originale."""
    
    # Estrai la loss e la MEE
    train_mse = np.array(history.history['mse'])
    val_mse = np.array(history.history['val_mse'])
    train_mee = np.array(history.history['mean_euclidean_error'])
    val_mee = np.array(history.history['val_mean_euclidean_error'])


    train_mse = original_scale(train_mse, scaler)
    val_mse = original_scale(val_mse, scaler)
    train_mee = original_scale(train_mee, scaler)
    val_mee = original_scale(val_mee, scaler)
    # Creazione del grafico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot delle perdite (MSE)
    ax1.plot(train_mse, label='Training MSE')
    ax1.plot(val_mse, label='Validation MSE')
    ax1.set_title('Training and Validation MSE (Original Scale)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.legend()
    ax1.grid()

    # Plot della Mean Euclidean Error (MEE)
    ax2.plot(train_mee, label='Training MEE')
    ax2.plot(val_mee, label='Validation MEE')
    ax2.set_title('Training and Validation MEE (Original Scale)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Euclidean Error')
    ax2.legend()
    ax2.grid()

    # Salva il grafico
    plt.tight_layout()
    plt.savefig(f'{path}/training_history_CUP_with_mee_original_scale.png')
    plt.close()


def plot_training_history_Monk(history, path):
    train_accuracies = history.history['accuracy']
    test_accuracies = history.history['val_accuracy']
    train_mse = history.history['mse']
    test_mse = history.history['val_mse']

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(train_accuracies) + 1)

    # Accuratezza
    ax1.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
    ax1.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Accuratezza')
    ax1.set_title('Accuratezza vs Epoche')
    ax1.legend()
    ax1.grid(True)

    # Mean Squared Error
    ax2.plot(epochs, train_mse, 'b-', label='Train MSE')
    ax2.plot(epochs, test_mse, 'r-', label='Test MSE')
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('MSE vs Epoche')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f'{path}/training_history_MONK.png')
    plt.close()

    print(f'Accuratezza Train finale: {train_accuracies[-1]:.4f}')
    print(f'Accuratezza Test finale: {test_accuracies[-1]:.4f}')
    print(f'MSE Train finale: {train_mse[-1]:.4f}')
    print(f'MSE Test finale: {test_mse[-1]:.4f}')