import matplotlib.pyplot as plt
import pandas as pd
def plot_training_history(history, window_size=5):
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
