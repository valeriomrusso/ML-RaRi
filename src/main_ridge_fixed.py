from load_data import load_and_preprocess_data_CUP
from plots import plot_training_history_CUP
from model_builder import build_model_ridge_fixed
from model_training import train_model_ridge_fixed
import pandas as pd

def main():
    filepath = 'ML-CUP24-TR.csv'
    reg = 0.00005
    learning_rate = 0.002
    momentum = 0.9
    batch_size = 80

    X_train, X_test, Y_train, Y_test, _, _ = load_and_preprocess_data_CUP(filepath)
    final_model = build_model_ridge_fixed(reg, learning_rate, momentum)
    history = train_model_ridge_fixed(final_model, batch_size, X_train, X_test, Y_train, Y_test)
    plot_training_history_CUP(history)

    final_dict = {}
    final_dict['batch_size'] = batch_size
    final_dict['reg'] = reg
    final_dict['learning_rate'] = learning_rate
    final_dict['momentum'] = momentum
    final_dict['train_loss'] = history.history['loss'][-1]
    final_dict['test_loss'] = history.history['val_loss'][-1]
    final_model_df = pd.DataFrame(final_dict, index=[0])
    final_model_df.to_csv('best_hps_model_fixed.csv', index=False)


if __name__ == "__main__":
    main()
