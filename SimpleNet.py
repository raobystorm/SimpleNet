
import numpy as np


class FullyConnectedLayer:
    # Usually the weight of fc layer is decided by the input & output's hidden units
    def __init__(self, input, output_size, activation, drop_rate=1.0):
        # Consider input from a conv layer
        self.input = input
        self.lower_layer = None
        self.output = np.zeros(output_size)
        self.weight =np.random.uniform(-0.05, 0.05, (self.input.size, self.output.size)) + 0.01
        self.delta = np.zeros(self.output.size)
        self.activation = activation
        # Used for drop-out, drop_rate means the proportion of weights we want to use
        self.drop_rate = drop_rate
        self.dropped_mask = np.ones(self.weight.shape, dtype=bool)


    def forward(self):
        if self.drop_rate < 1.0:
            self.dropped_mask = np.random.uniform(0, 1, self.output.shape)
            self.dropped_mask = np.where(self.dropped_mask <= self.drop_rate, 1, 0)

        y = np.dot(self.input, self.weight * self.dropped_mask)

        if self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-y))
        if self.activation == 'softmax':
            exp = np.exp(y)
            self.output = exp / float(sum(exp))


    def backward(self, label=None):
        import pdb
        pdb.set_trace()
        if self.activation == 'sigmoid':
            self.delta += np.dot(self.lower_layer.delta, self.lower_layer.weight.T) * (self.output * (1 - self.output))
        # For output layer only, we use label
        # The loss function is cross entropy loss
        if self.activation == 'softmax':
            self.delta += self.output - label

    def update_weight(self, learn_rate=0.05):
        self.weight -= self.input.T.dot(self.delta * self.dropped_mask) * learn_rate


class Network:
    def __init__(self, layer_sizes, learn_rate=0.05):
        self.input = np.zeros(layer_sizes[0], dtype=np.float32)
        self.learn_rate = learn_rate
        self.layers = []
        last_input = self.input
        for i in range(len(layer_sizes)):
            if i == len(layer_sizes) - 1:
                fc = FullyConnectedLayer(last_input, layer_sizes[i], 'softmax', drop_rate=0.5)
            else :
                fc = FullyConnectedLayer(last_input, layer_sizes[i], 'sigmoid', drop_rate=0.5)
            last_input = fc.output
            self.layers.append(fc)

        last_layer = self.layers[-1]
        for layer in reversed(self.layers[:-1]):
            layer.lower_layer = last_layer
            last_layer = layer
        self.output = self.layers[-1].output


    def inference(self, data):
        self.layers[0].input = data
        for layer in self.layers:
            layer.forward()


    def train(self, train_set, epochs=10, batch_size=30):
        for _ in range(epochs):
            np.random.shuffle(train_set)
            for iter in range(len(train_set) // batch_size):
                self.loss = 0.0
                correct = 0
                for data in train_set[ iter*batch_size : (iter + 1)*batch_size]:
                    self.layers[0].input = data[:-1]
                    for layer in self.layers:
                        layer.forward()
                    for layer in reversed(self.layers):
                        layer.backward(data[-1])
                    self.loss += -(data[-1] * np.log(self.output))
                    # binarize output and minus the label to see if prediction is correct
                    if sum(data[-1] - np.where(np.max(self.output), 1., 0.)) == 0:
                        correct += 1

                print('Current loss is: ' + str(self.loss))
                print('Current accuracy is: ' + str(correct / batch_size))
                for layer in self.layers:
                    layer.update_weight(self.learn_rate)


def label_iris(train_set):
    label_set = [data[-1] for data in train_set]
    label_set = list(set(label_set))
    for data in train_set:
        idx = label_set.index(data[-1])
        data[-1] = np.zeros(len(label_set))
        data[-1][idx] = 1


def run(args):
    train_set = []
    with open(args.input, 'r') as f:
        for line in f.readlines():
            data = list(map(float, line.split(',')[:-1]))
            label = line.split(',')[-1]
            data .append(label)
            train_set.append(data)

    label_iris(train_set)

    net = Network([4, 3])
    net.train(train_set, epochs=10, batch_size=30)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='iris.data file', required=True)
    args = parser.parse_args()

    return args

run(parse_args())