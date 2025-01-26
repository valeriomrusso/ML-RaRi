import pandas as pd

def csv_builder(path, dicthps):
    df = pd.DataFrame(dicthps, index=[0])
    df.to_csv(path, index=False)