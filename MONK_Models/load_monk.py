import numpy as np
from sklearn.model_selection import train_test_split

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
    train_data = load_monk_data(f"{monk}.train")
    test_data = load_monk_data(f"{monk}.test")

    # Extract features and target
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test