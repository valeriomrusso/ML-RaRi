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
    mean_scale = scaler.scale_.mean()
    
    # Riporta il MEE alla scala originale
    original_data = scaled_data * mean_scale
    
    return original_data
