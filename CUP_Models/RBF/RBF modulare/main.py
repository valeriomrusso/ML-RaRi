from load_data import load_and_preprocess_data_CUP
from plots import plot_training_history_CUP
from cross_validation_rbf import CV

def main():
    filepath = 'ML-CUP24-TR.csv'
    X_train, X_test, Y_train, Y_test, _, _ = load_and_preprocess_data_CUP(filepath)
    final_history, final_model = CV(X_train, X_test, Y_train, Y_test, "CUP", "RBF")
    plot_training_history_CUP(final_history)

if __name__ == "__main__":
    main()