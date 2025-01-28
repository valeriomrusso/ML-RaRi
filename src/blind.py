import pandas as pd
import numpy as np
from load_data import load_blind_test
from model_builder import build_model_nn_fixed
from build_csv import writeOutput

def main():
    X, Y, X_test, scaler = load_blind_test('./datasets/ML-CUP24-TR.csv', './datasets/ML-CUP24-TS.csv')
    model = build_model_nn_fixed(32,0.0, 2, 0.0037901949810414497,8.575698119716961e-06,'CUP')
    model.fit(X, Y, shuffle = True, epochs=500, batch_size=12)
    PD = model.predict(X_test)
    PD = scaler.inverse_transform(PD)
    writeOutput(PD, "./datasets/Risi_Scotti_ML-CUP24-TS.csv")

if __name__ == "__main__":
    main()