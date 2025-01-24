from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_cup_data():
    np.set_printoptions(precision=20, suppress=True)

    my_data = np.genfromtxt('ML-CUP24-TR.csv', delimiter=',')
    X = my_data[:, 1:13]
    y = my_data[:, 13:16]
    print(X.shape, y.shape)

    #print(X)
    #print(y)

    # Suddividi i dati in train (60%) e temp (40%) (HOLDOUT)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    # Suddividi temp in validation (20%) e test (20%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Normalizza i dati
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train = scaler_X.transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test