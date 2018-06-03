
import numpy as np


class FullyConnectedLayer:
    # Usually the weight of fc layer is decided by the input & output's hidden units
    def __init__(self, input, output_size, activation, drop_rate=1.0):
        # Consider input from a conv layer
        self.input = input
        self.output = np.random.random(output_size)
        self.weight = np.random.random((input.size, output_size,))
        self.bias = np.random.random(output_size)
        self.delta = np.zeros(output_size)
        self.activation = activation
        # Used for drop-out, drop_rate means the proportion of weights we want to use
        self.drop_rate = drop_rate
        self.dropped_mask = np.ones(self.weight.shape, dtype=bool)


    def forward(self):
        if self.drop_rate < 1.0:
            self.dropped_mask = np.random.uniform(0, 1, self.output.shape)
            self.dropped_mask = np.where(self.dropped_mask <= self.drop_rate, 1, 0)
        if self.activation == 'sigmoid':
            y = np.dot(self.input, self.weight * self.dropped_mask) + self.bias
            self.output = 1 / (1 + np.exp(-y))
        if self.activation == 'softmax':
            y = np.dot(self.input, self.weight * self.dropped_mask) + self.bias
            self.output = np.exp(y) / float(sum(np.exp(y)))


    def backward(self, label=None, learn_rate=0.05):
        if self.activation == 'sigmoid':
            self.delta = (self.output - self.input) * (self.input * (1 - self.input))
            if self.drop_rate < 1.0:
                self.dropped_mask = np.random.
                self.weight += self.input.T.dot(self.delta) * learn_rate
        # For output layer only, we use label
        # The loss function is sum((label-output) * ln(output))
        if self.activation == 'softmax':
            self.delta = label - self.output
            self.weight -= self.input.T.dot(self.delta) * learn_rate


class Network:
    def __init__(self, layer_sizes, learn_rate=0.05):
        self.input = np.zeros(layer_sizes[0], dtype=np.float32)
        self.learn_rate = learn_rate
        self.layers = []
        last_input = self.input
        for i in range(1, len(layer_sizes)):
            if i == len(layer_sizes) - 1:
                fc = FullyConnectedLayer(last_input, layer_sizes[i], 'softmax', drop_rate=0.5)
            else :
                fc = FullyConnectedLayer(last_input, layer_sizes[i], 'sigmoid', drop_rate=0.5)
            last_input = fc.output
            self.layers.append(fc)
        self.output = self.layers[-1].output


    def inference(self, data):
        self.layers[0].input = data
        for layer in self.layers:
            layer.forward()


    def train(self, train_set):
        for data, label in train_set:
            self.layers[0].input = data
            for layer in self.layers:
                layer.forward()
            for layer in reversed(self.layers):
                layer.backward(label, learn_rate=self.learn_rate)


def run(args):
    train_set = []
    with open(args.input, 'r') as f:
        for line in f:
            data = list(map(float, line.split(',')[:-1]))
            data += line.split(',')[-1]
            train_set.append(data)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='iris.data file', required=True)
    parser.add_argument('--output', type=str, help='output dir', required=True)
    args = parser.parse_args()

    return args

run(parse_args())