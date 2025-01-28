from nn import NN
from ridge import Ridge

def main():
    dropout = 0.1
    num_layers = 7
    units = 96
    learning_rate = 0.0007389159591781176
    momentum = 0.9
    reg = 0.00038129678768194814
    batch_size = 24
    # Task, nmonk, fixed(True, false), tunertype, units, dropout, num_layers, units_hidden, learning_rate, momentum, reg, batch_size
    #NN("CUP", 1, True, units, dropout, num_layers, learning_rate, momentum, reg, batch_size)
    NN('CUP', 2, False)
if __name__ == '__main__':
    main()