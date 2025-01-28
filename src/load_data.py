import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data_CUP(filepath):
    """Carica e normalizza i dati da un file CSV."""
    my_data = np.genfromtxt(filepath, delimiter=',')
    X = my_data[:, 1:13]
    Y = my_data[:, 13:16]
    
    scalerX = StandardScaler().fit(X)
    scalerY = StandardScaler().fit(Y)
    
    X = scalerX.transform(X)
    Y = scalerY.transform(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test, scalerX, scalerY

# Function to load the dataset
def load_monk_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ignore empty lines
                parts = line.strip().split()  # Split by spaces
                target = int(parts[0])  # First column is the target
                features = list(map(int, parts[1:7]))  # Next six columns are features
                data.append([target] + features)
    return np.array(data)


def splitted_monk_data(monk):
    # Load train and test datasets
    train_data = load_monk_data(f"./datasets/{monk}.train")
    test_data = load_monk_data(f"./datasets/{monk}.test")

    # Extract features and target
    X_train = train_data[:, 1:]
    Y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    Y_test = test_data[:, 0]

    # One-Hot Encoding
    encoder = OneHotEncoder()
    X_train_enc = encoder.fit_transform(X_train).toarray()
    X_test_enc = encoder.fit_transform(X_test).toarray()
    
    scalerX = StandardScaler().fit(X_train_enc)
    X_train_enc = scalerX.transform(X_train_enc)
    X_test_enc = scalerX.transform(X_test_enc)


    return X_train_enc, X_test_enc, Y_train, Y_test

def load_blind_test(train, test):
    train_data = np.genfromtxt(train, delimiter=',')
    X = train_data[:, 1:13]
    Y = train_data[:, 13:16]
    scalerX = StandardScaler().fit(X)
    scalerY = StandardScaler().fit(Y)
    X = scalerX.transform(X)
    Y = scalerY.transform(Y)
    test_data = np.genfromtxt(test, delimiter=',')
    X_test = test_data[:, 1:13]
    X_test = scalerX.transform(X_test)
    return X, Y, X_test, scalerY
    