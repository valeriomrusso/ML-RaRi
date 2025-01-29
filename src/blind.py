# Import necessary libraries
import pandas as pd
import numpy as np
from load_data import load_blind_test  
from model_builder import build_model_nn_fixed 
from build_csv import writeOutput  

def main():
    X, Y, X_test, scaler = load_blind_test('./datasets/ML-CUP24-TR.csv', './datasets/ML-CUP24-TS.csv')
    
    # Build a fixed neural network model with specified hyperparameters
    model = build_model_nn_fixed(32, 0.0, 3, 0.0022232368862489644, 5.026876136708517e-06, 'CUP')
    
    # Train the final model on the whole TR dataset
    model.fit(X, Y, shuffle=True, epochs=500, batch_size=112)
    
    # Predict on the blind test dataset
    PD = model.predict(X_test)
    
    # Reverse the normalization of predictions
    PD = scaler.inverse_transform(PD)
    
    # Save the predictions to a CSV file
    writeOutput(PD, "./datasets/Risi_Scotti_ML-CUP24-TS.csv")

if __name__ == "__main__":
    main()
