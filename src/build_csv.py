import pandas as pd
import numpy as np
'''
def csv_builder_CUP(path, dicthps):
    df = pd.DataFrame(dicthps)
    df.to_csv(path, index=False)
'''
def csv_builder(path, dicthps):
    # Se dicthps è un singolo dizionario, fai un elenco con esso
    if isinstance(dicthps, dict):
        dicthps = [dicthps]
    
    # Ora pandas può creare un DataFrame, poiché è una lista di dizionari
    df = pd.DataFrame(dicthps)

    # Salva il CSV
    df.to_csv(path, index=False)

import numpy as np

def original_scale(scaled_data, scaler):
    # Se il dato è un singolo valore (scalato), lo trasformiamo in un array 2D
    if isinstance(scaled_data, (float, int)):  # Controlliamo se è un singolo valore
        scaled_data = np.array([[scaled_data]])  # Convertiamolo in un array 2D
    
    # Verifica se scaled_data è un array con una sola colonna
    if len(scaled_data.shape) == 1:
        scaled_data = scaled_data.reshape(-1, 1)  # Ristrutturazione a colonna singola
    
    # Ora, supponendo che scalerY sia stato adattato su 3 colonne di output,
    # dobbiamo fare in modo che l'array abbia 3 colonne, anche se il MSE è un singolo valore
    if scaled_data.shape[1] != scaler.scale_.shape[0]:  # Se il numero di colonne è diverso da quello dello scaler
        # Ridimensiona l'array a 3 colonne
        scaled_data = np.repeat(scaled_data, scaler.scale_.shape[0], axis=1)
    
    # Inverso del scaling
    return scaler.inverse_transform(scaled_data).flatten()