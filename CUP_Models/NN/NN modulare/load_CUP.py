import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    """Carica e normalizza i dati da un file CSV."""
    my_data = np.genfromtxt(filepath, delimiter=',')
    X = my_data[:, 1:13]
    Y = my_data[:, 13:16]
    
    scalerX = StandardScaler().fit(X)
    scalerY = StandardScaler().fit(Y)
    
    X = scalerX.transform(X)
    Y = scalerY.transform(Y)
    
    return X, Y, scalerX, scalerY