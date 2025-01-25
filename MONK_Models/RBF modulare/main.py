from load_data import load_and_preprocess_data_MONK
from plots import plot_training_history_Monk
from cross_validation_rbf import CV

def main():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_and_preprocess_data_MONK("monks-1")
    final_history, final_model = CV(X_train, X_test, Y_train, Y_test, "MONK", "RBF")
    plot_training_history_Monk(final_history)

if __name__ == "__main__":
    main()