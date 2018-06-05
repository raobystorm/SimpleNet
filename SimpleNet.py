
import numpy as np


class FullyConnectedLayer:
    # Usually the weight of fc layer is decided by the input & output's hidden units
    def __init__(self, output_size, activation, input_size=0, upper_layer=None, drop_rate=1.0):
        # Consider input from a conv layer
        self.upper_layer = upper_layer
        if upper_layer:
            input_size = upper_layer.output.size
        self.input = None
        self.lower_layer = None
        self.output = np.zeros(output_size)
        self.weight = np.random.uniform(-0.05, 0.05, (input_size, self.output.size)) + 0.01
        self.delta = np.zeros(self.output.size)
        self.activation = activation
        # Used for drop-out, drop_rate means the proportion of weights we want to use
        self.drop_rate = drop_rate
        self.dropped_mask = np.ones(self.weight.shape, dtype=bool)


    def forward(self):
        if self.upper_layer:
            self.input = self.upper_layer.output
        if self.drop_rate < 1.0:
            self.dropped_mask = np.random.uniform(0, 1, self.weight.shape)
            self.dropped_mask = np.where(self.dropped_mask <= self.drop_rate, 1, 0)

        y = np.dot(self.input, self.weight * self.dropped_mask)

        if self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-y))
        if self.activation == 'softmax':
            exp = np.exp(y)
            self.output = exp / float(sum(exp))


    def backward(self, label=None):
        if self.activation == 'sigmoid':
            self.delta += np.dot(self.lower_layer.weight, self.lower_layer.delta) * (self.output * (1 - self.output))
        # For output layer only, we use label
        # The loss function is cross entropy loss
        if self.activation == 'softmax':
            self.delta -= self.output - label


    def update_weight(self, learn_rate=0.05):
        self.weight += np.outer(self.input, self.delta) * self.dropped_mask * learn_rate


class Network:
    def __init__(self, layer_sizes, learn_rate=0.03, loss_func='cross-entropy'):
        self.input = np.zeros(layer_sizes[0], dtype=np.float32)
        self.learn_rate = learn_rate
        self.layers = []
        self.loss_func = loss_func
        last_layer = None
        for i in range(1, len(layer_sizes)):
            if i == len(layer_sizes) - 1:
                fc = FullyConnectedLayer(layer_sizes[i], 'softmax', input_size=self.input.size, upper_layer=last_layer, drop_rate=1.)
            else:
                fc = FullyConnectedLayer(layer_sizes[i], 'sigmoid', input_size=self.input.size, upper_layer=last_layer, drop_rate=0.5)
            last_layer = fc
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
        return self.layers[-1].output


    def train(self, data, label):
        prediction = self.inference(data)
        for layer in self.layers:
            layer.backward(label)
        if self.loss_func == 'cross-entropy':
            self.loss += -sum(label * np.log(prediction))
        self.output = prediction

    def save_weights(self):
        self.saved_weights = []
        for layer in self.layers:
            self.saved_weights.append(np.copy(layer.weight))


    def load_weights(self):
        if not self.saved_weights:
            return
        for idx, saved_weight in enumerate(self.saved_weights):
            self.layers[idx].weight = saved_weight

    def is_prediction_correct(self, label):
        return sum(abs(label - np.where(self.output == np.max(self.output), 1., 0.))) == 0


    def test_set(self, test_set):
        correct = 0
        for test_point in test_set:
            result = self.inference(test_point[:-1])
            if self.is_prediction_correct(test_point[:-1]):
                correct += 1
        return correct / len(test_set)


    def train_set(self, train_set, epochs=10, batch_size=50):
        max_acc = 0.0
        for _ in range(epochs):
            np.random.shuffle(train_set)
            for iter in range(len(train_set) // batch_size):
                self.loss = 0.0
                correct = 0
                for layer in self.layers:
                    layer.delta = np.zeros(layer.delta.shape, dtype=np.float32)
                for data in train_set[ iter*batch_size : (iter + 1)*batch_size]:
                    self.train(data[:-1], data[-1])
                    # binarize output and minus the label to see if prediction is correct
                    if self.is_prediction_correct(data[-1]):
                        correct += 1

                print('Current loss is: ' + str(self.loss))
                acc = correct / batch_size
                if acc > max_acc:
                    max_acc = acc
                    self.save_weights()

                print('Current accuracy is: ' + str(correct / batch_size))
                for layer in self.layers:
                    layer.update_weight(self.learn_rate)
                print('Lear rate is:' + str(self.learn_rate))
                print('')


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
            data.append(label)
            train_set.append(data)

    label_iris(train_set)

    # For the layer size first one is input size, last one is the label vector size.
    net = Network([4, 3], learn_rate=0.03)
    net.train_set(train_set, epochs=10, batch_size=10)

    print('Finished training! Start testing...')

    net.load_weights()
    acc = net.test_set(np.random.choice(train_set, 30))
    print('Finished testing! The result accuracy is: ' + str(acc))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='iris.data file', required=True)
    args = parser.parse_args()

    return args

run(parse_args())