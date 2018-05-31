
import numpy as np

class FullyConnectedLayer:
    def __init__(self, input, output):
        input_image_size = input.shape[0] * input.shape[1]
        output_image_size = output.shape[0] * output.shape[1]
        self.weight = np.Random.random(input_image_size * output_image_size, 1, )
        self.bias = np.Random.random()
        self.input = np.reshape(input, (input_image_size, 1, input.shape[3], input.shape[4]))
        self.output = np.reshape(output, (output_image_size, 1, input.shape[3], input.shape[4]))

    def forward(self):

    def backward(self):