import pandas as pd
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