from load_data import *
from plots import plot_training_history_CUP
from cross_validation import CV

def main():
    task = 'CUP'
    
    if task == 'CUP':
        filepath = './datasets/ML-CUP24-TR.csv'
        X_train, X_test, Y_train, Y_test, _, _ = load_and_preprocess_data_CUP(filepath)
    elif task == 'MONK':
        for monk in ['monks-1', 'monks-2', 'monks-3']:
            filepath = monk
            X_train, X_test, Y_train, Y_test = splitted_monk_data(filepath)

    final_history, final_model = CV(X_train, X_test, Y_train, Y_test, task, "NN")
    plot_training_history_CUP(final_history)


if __name__ == "__main__":
    main()
