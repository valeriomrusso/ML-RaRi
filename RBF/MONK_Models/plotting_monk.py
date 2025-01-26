import matplotlib.pyplot as plt

def plot_history(history):
    # Plot training and test accuracy
    plt.figure(figsize=(5, 3))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], linestyle='dotted', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot training and test MSE
    plt.figure(figsize=(5, 3))
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], linestyle='dotted', label='Test MSE')
    plt.title('Training and Test MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calcolo delle metriche
    tr_accuracy = max(history.history['accuracy'])
    val_accuracy = max(history.history['val_accuracy'])
    tr_mse = min(history.history['mse'])
    val_mse = min(history.history['val_mse'])
    print(f"Training Accuracy: {tr_accuracy}")
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Training MSE: {tr_mse}")
    print(f"Validation MSE: {val_mse}") 