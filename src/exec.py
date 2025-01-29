from nn import NN
from ridge import Ridge

def main():
    # Define hyperparameters
    dropout = 0.1
    num_layers = 7
    units = 96
    learning_rate = 0.0007389159591781176
    momentum = 0.9
    reg = 0.00038129678768194814
    batch_size = 24
    
    # Run the NN model for the CUP task with predefined configurations (nmonk = 2, fixed = False)
    NN('CUP', 2, False)
if __name__ == '__main__':
    main()