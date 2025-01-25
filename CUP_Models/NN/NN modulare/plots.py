import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
def plot_training_history_CUP(history, window_size=5):
    """Visualizza l'andamento del training."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot delle perdite grezze
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid()
    
    # Plot delle perdite smussate
    smoothed_train = pd.Series(history.history['loss']).rolling(window=window_size).mean()
    smoothed_val = pd.Series(history.history['val_loss']).rolling(window=window_size).mean()
    ax2.plot(smoothed_train, 'r-', label=f'Smoothed Training Loss (window={window_size})')
    ax2.plot(smoothed_val, 'b-', label=f'Smoothed Validation Loss (window={window_size})')
    ax2.set_title('Smoothed Training and Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_training_history_Monk(history, model, X_train, y_train, X_test, y_test):
    train_accuracies = history.history['accuracy']
    test_accuracies = history.history['val_accuracy']
    train_losses = history.history['loss']
    test_losses = history.history['val_loss']

    # Compute mean squared error
    train_mse = [mean_squared_error(y_train, model.predict(X_train))]*len(train_accuracies)
    test_mse = [mean_squared_error(y_test, model.predict(X_test))]*len(test_accuracies)

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
    plt.savefig('training_history.png')
    plt.show()

    print(f'Accuratezza Train finale: {train_accuracies[-1]:.4f}')
    print(f'Accuratezza Test finale: {test_accuracies[-1]:.4f}')
    print(f'MSE Train finale: {train_mse[-1]:.4f}')
    print(f'MSE Test finale: {test_mse[-1]:.4f}')