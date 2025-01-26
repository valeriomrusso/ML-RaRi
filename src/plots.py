import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
def plot_training_history_CUP(history, path, window_size=5):
    """Visualizza l'andamento del training."""
    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 5))
    
    # Plot delle perdite grezze
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid()
    
    plt.tight_layout()
    plt.savefig(f'{path}/training_history_CUP.png')
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

    plt.suptitle('Performance Rete Neurale MONK-1')
    plt.tight_layout()
    plt.savefig(f'{path}/training_history_MONK.png')
    plt.close()

    print(f'Accuratezza Train finale: {train_accuracies[-1]:.4f}')
    print(f'Accuratezza Test finale: {test_accuracies[-1]:.4f}')
    print(f'MSE Train finale: {train_mse[-1]:.4f}')
    print(f'MSE Test finale: {test_mse[-1]:.4f}')