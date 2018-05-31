
import numpy as np

class FullyConnectedLayer:
    # Usually the weight of fc layer is decided by the input & output's hidden units
    def __init__(self, input, output, activation):
        input_size = input.shape[0] * input.shape[1]
        output_size = output.shape[0]
        self.input = np.reshape(input, (input_size, 1,))
        self.output = np.reshape(output, (output_size, 1,))
        self.weight = np.random.random(input_size, output_size,)
        self.bias = np.random.random(output_size, 1,)
        self.delta = np.random.random(output.shape)
        self.activation = activation

    def forward(self):
        if self.activation == 'logistic':
            self.output = 1 / (1 + np.exp(-np.dot(self.input, self.weight)))

    def backward(self):
        if self.activation == 'logistic':
            self.delta = (self.output - self.input) * (self.input * (1 - self.input))
            self.weight += self.input.T.dot(self.delta)