import pandas as pd
import numpy as np
import datetime
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

def writeOutput(result, name):
        df = pd.DataFrame(result)
        now = datetime.datetime.now()
        f = open(name, 'w')
        f.write('# Michele Di Niccola, Valerio Russo\n')
        f.write('# team-name\n')
        f.write('# ML-CUP24\n')
        f.write('# '+str(now.day)+'/'+str(now.month)+'/'+str(now.year)+'\n')
        df.index += 1 
        df.to_csv(f, sep=',', encoding='utf-8', header = False)
        f.close()
