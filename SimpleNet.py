
import numpy as np


class FCLayer:
    # Usually the weight of fc layer is decided by the input & output's hidden units
    def __init__(self, output_size, input_size=0, upper_layer=None, drop_rate=1.0):
        # Consider input from a conv layer
        self.upper_layer = upper_layer
        if upper_layer:
            input_size = upper_layer.output.size
        self.input = None
        self.lower_layer = None
        self.output = np.zeros(output_size)
        # self.weight = np.random.randn(input_size, output_size) * np.sqrt(2 / ((input_size) * drop_rate))
        self.weight = np.random.uniform(-0.001, 0.001, (input_size, output_size))
        self.delta = np.zeros(self.output.size)
        # Used for drop-out, drop_rate means the proportion of weights we want to use
        self.drop_rate = drop_rate
        self.dropped_mask = np.ones(self.weight.shape, dtype=bool)

    def forward(self):
        if self.upper_layer:
            self.input = self.upper_layer.output
        if self.drop_rate < 1.0:
            self.dropped_mask = np.random.uniform(0, 1, self.weight.shape)
            self.dropped_mask = np.where(self.dropped_mask <= self.drop_rate, 1, 0)
        self.output = np.dot(self.input, self.weight * self.dropped_mask)

    def backward(self):
        raise NotImplementedError

    def update_weight(self, learn_rate=0.05):
        self.weight -= np.outer(self.input, self.delta) * self.dropped_mask * learn_rate


class FullyConnectedLayerSigmoid(FCLayer):
    def forward(self):
        super(FullyConnectedLayerSigmoid, self).forward()
        # self.output = 1 / (1 + np.exp(-self.output))

    def backward(self, label=None):
        if self.lower_layer:
            self.delta += np.dot(self.lower_layer.weight, self.lower_layer.delta) * self.output * (1 - self.output)
        else:
            self.delta += self.output - label


class SoftmaxLayer(FCLayer):
    def forward(self):
        super(SoftmaxLayer, self).forward()
        exp = np.exp(self.output)
        self.output = exp / sum(exp)

    def backward(self, label=None):
        self.delta += self.output - label


class FullyConnectedLayerReLU(FCLayer):
    def forward(self):
        super(FullyConnectedLayerReLU, self).forward()
        self.dropped_mask = np.where(self.output < 0, False, True)
        self.output = np.where(self.output < 0, 0., self.output)

    def backward(self):
        self.delta += np.dot(self.lower_layer.weight, self.lower_layer.delta)


def cross_entropy_func(prediction, label):
    return -sum(label * np.log(prediction))


def sum_of_squares_func(prediction, label):
    return sum((prediction - label)**2) / 2


class Network:
    def __init__(self, layer_sizes, learn_rate=0.03, loss_func=cross_entropy_func, log_file=None):
        if log_file:
            self.log = open(log_file, 'w')
        self.input = np.zeros(layer_sizes[0], dtype=np.float32)
        self.learn_rate = learn_rate
        self.layers = []
        self.loss_func = loss_func
        last_layer = None
        for i in range(1, len(layer_sizes)):
            if i == len(layer_sizes) - 1:
                fc = SoftmaxLayer(layer_sizes[i], input_size=self.input.size, upper_layer=last_layer, drop_rate=1.)
            else:
                fc = FullyConnectedLayerSigmoid(layer_sizes[i], input_size=self.input.size, upper_layer=last_layer, drop_rate=1.)
            last_layer = fc
            self.layers.append(fc)

        last_layer = self.layers[-1]
        for layer in reversed(self.layers[:-1]):
            layer.lower_layer = last_layer
            last_layer = layer
        self.output = self.layers[-1].output

    def __del__(self):
        self.log.close()

    def inference(self, data):
        self.layers[0].input = data
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].output

    def train(self, data, label):
        prediction = self.inference(data)
        self.layers[-1].backward(label)
        for layer in self.layers[:-1]:
            layer.backward()
        self.output = prediction
        self.loss += self.loss_func.__call__(prediction, label)

    def save_weights(self):
        self.saved_weights = []
        for layer in self.layers:
            self.saved_weights.append(np.copy(layer.weight))

    def load_weights(self):
        if not self.saved_weights:
            return
        for idx, saved_weight in enumerate(self.saved_weights):
            self.layers[idx].weight = saved_weight

    def print_weights(self):
        for layer in self.layers:
            print(layer.weight)

    def is_prediction_correct(self, label):
        return np.array_equal(np.where(self.output == np.max(self.output), 1., 0.), label)

    def test_set(self, test_set):
        correct = 0
        self.reset_drop_mask()
        for test_point in test_set:
            self.output = self.inference(test_point[:-1])
            if self.is_prediction_correct(test_point[-1]):
                correct += 1
        return correct / len(test_set)

    def reset_drop_mask(self):
        for layer in self.layers:
            layer.dropped_mask = np.ones(layer.dropped_mask.shape)

    def train_set(self, train_set, epochs=10, batch_size=50):
        min_loss = 999999.
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

                self.log.write(str(self.loss) + ',' + str(correct / batch_size) + '\n')
                if self.loss < min_loss:
                    min_loss = self.loss
                    self.save_weights()

                for layer in self.layers:
                    layer.update_weight(self.learn_rate)


def label_iris(train_set):
    label_set = [data[-1] for data in train_set]
    label_set = sorted(list(set(label_set)))
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
    net = Network([4, 4, 3], learn_rate=0.01, loss_func=cross_entropy_func, log_file=args.log)
    net.train_set(train_set, epochs=500, batch_size=30)

    print('Finished training! Start testing...')

    net.load_weights()
    acc = net.test_set(train_set)
    print('Finished testing! The result accuracy is: ' + str(acc))
    net.print_weights()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='iris.data file', required=True)
    parser.add_argument('--log', type=str, help='log file', required=True)
    args = parser.parse_args()

    return args

run(parse_args())