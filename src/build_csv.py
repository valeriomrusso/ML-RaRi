import pandas as pd
import datetime

# Save a dictionary as a CSV file
def csv_builder(path, dicthps):
    if isinstance(dicthps, dict):
        dicthps = [dicthps]
    
    df = pd.DataFrame(dicthps)

    df.to_csv(path, index=False)

#Scale the data back to its original range
def original_scale(scaled_data, scaler):
    mean_scale = scaler.scale_.mean()
    
    original_data = scaled_data * mean_scale
    
    return original_data

#Writes the result to a CSV file with a custom header
def writeOutput(result, name):
        df = pd.DataFrame(result)
        now = datetime.datetime.now()
        f = open(name, 'w')
        f.write('# Michele Di Niccola, Valerio Russo\n')
        f.write('# Risi Scotti\n')
        f.write('# ML-CUP24\n')
        f.write('# '+str(now.day)+'/'+str(now.month)+'/'+str(now.year)+'\n')
        df.index += 1 
        df.to_csv(f, sep=',', encoding='utf-8', header = False)
        f.close()
