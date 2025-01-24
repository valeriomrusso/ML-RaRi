import matplotlib.pyplot as plt

def plot_cup(history):
    # Visualizzazione delle prestazioni
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    tr_loss = min(history.history['loss'])
    val_loss = min(history.history['val_loss'])
    print(f"Training Loss: {tr_loss}")
    print(f"Validation Loss: {val_loss}")