
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
        # self.weight = np.ones((input_size, output_size)) * np.sqrt(2 / (input_size)) + 0.1
        self.weight = np.random.randn(input_size, output_size) * np.sqrt(2 / ((input_size) * drop_rate))
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
        self.output = 1 / (1 + np.exp(-self.output))

    def backward(self, label=None):
        if self.lower_layer:
            self.delta += np.dot(self.lower_layer.weight, self.lower_layer.delta) * (self.output * (1 - self.output))
        else:
            self.delta += self.output - label


class SoftmaxLayer(FCLayer):
    def forward(self):
        super(SoftmaxLayer, self).forward()
        exp = np.exp(self.output)
        self.output = exp / sum(exp)

    def backward(self, label=None):
        self.delta += self.output - label


class COCOLossLayer(FCLayer):
    def __init__(self, output_size, input_size, upper_layer, drop_rate, centers):
        super(COCOLossLayer, self).__init__(output_size, input_size, upper_layer, drop_rate)
        self.centers = centers

    def forward(self):
        super(COCOLossLayer, self).forward()
        self.output = self.output / np.linalg.norm(self.output)
        deno = 0.
        for center in self.centers:
            deno += np.exp(np.dot(center.T, self.output))
        exp = []
        for center in sorted(self.centers.keys()):
            exp.append(np.exp(np.dot(self.centers[center].T, self.output)))
        self.output = np.asarray(exp)

    def backward(self, label=None):
        update_delta = []
        for center in sorted(self.centers.keys()):
            update_delta.append(np.dot(self.output - label, center))
        self.delta += update_delta


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


def congenerous_cosine_similarity(ci, cj):
    ci = np.asarray(ci, dtype=np.float32)
    cj = np.asarray(cj, dtype=np.float32)
    return np.dot(ci.T, cj) / (np.linalg.norm(ci) * np.linalg.norm(cj))


def congenerous_cosine_func(prediction, label):
    return -sum(label * np.log(prediction))


class Network:
    def __init__(self, layer_sizes, learn_rate=0.03, loss_func=cross_entropy_func, centers=None):
        self.input = np.zeros(layer_sizes[0], dtype=np.float32)
        self.learn_rate = learn_rate
        self.layers = []
        self.train_log = []
        self.loss = 0.0
        self.loss_func = loss_func
        last_layer = None
        for i in range(1, len(layer_sizes)):
            # For this is the output layer, use activation according to their loss function
            if i == len(layer_sizes) - 1:
                if loss_func == cross_entropy_func:
                    fc = SoftmaxLayer(layer_sizes[i], input_size=self.input.size, upper_layer=last_layer)
                elif loss_func == sum_of_squares_func:
                    fc = FullyConnectedLayerSigmoid(layer_sizes[i], input_size=self.input.size, upper_layer=last_layer)
                elif loss_func == congenerous_cosine_func:
                    fc = COCOLossLayer(layer_sizes[i], input_size=self.input.size, upper_layer=last_layer, centers=centers)
            else:
                fc = FullyConnectedLayerSigmoid(layer_sizes[i], input_size=self.input.size, upper_layer=last_layer, drop_rate=1.)
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
        self.layers[-1].backward(label)
        for layer in self.layers[:-1]:
            layer.backward()
        self.output = prediction
        self.loss += self.loss_func.__call__(prediction, label)

    def update_weights(self):
        for layer in self.layers:
            layer.update_weight(self.learn_rate)

    def reset_deltas(self):
        for layer in self.layers:
            layer.delta = np.zeros(layer.output.shape)

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
            print(str(layer))
            print(layer.weight)

    def print_layer_outputs(self):
        for layer in self.layers:
            print(str(layer))
            print(layer.output)

    def print_deltas(self):
        for layer in self.layers:
            print(str(layer))
            print(layer.delta)

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
        self.train_log = []
        if self.loss_func == congenerous_cosine_func:
            loss_center = compute_coco_center(train_set)

        for _ in range(epochs):
            np.random.shuffle(train_set)
            for iter in range(len(train_set) // batch_size):
                self.loss = 0.0
                correct = 0
                self.reset_deltas()
                for data in train_set[ iter*batch_size : (iter + 1)*batch_size]:
                    self.train(data[:-1], data[-1])
                    # binarize output and minus the label to see if prediction is correct
                    if self.is_prediction_correct(data[-1]):
                        correct += 1

                self.train_log.append(self.loss)
                if self.loss < min_loss:
                    min_loss = self.loss
                    self.save_weights()
                self.update_weights()


    def evaluate(self, input, epochs=500, batch_size=30):
        train_set, test_set = prepare_dataset_iris(input)
        self.train_set(train_set, epochs=epochs, batch_size=batch_size)
        print('Finished training! Start testing...')

        self.load_weights()
        acc = self.test_set(test_set)
        print('Finished testing! The result accuracy is: ' + str(acc))
        self.print_weights()


def compute_coco_center(train_set):
    clusters = {}
    centers = {}
    for data in train_set:
        if str(data[-1]) not in clusters:
            clusters[str(data[-1])] = []
        clusters[str(data[-1])].append(data[:-1])

    for str_c, cluster in clusters:
        raw_center = np.mean(cluster, axis=0)
        centers[str_c] = np.asarray(raw_center / np.linalg.norm(raw_center))

    return centers


def prepare_dataset_iris(input):
    train_set = []
    with open(input, 'r') as f:
        for line in f.readlines():
            data = list(map(float, line.split(',')[:-1]))
            label = line.split(',')[-1]
            data.append(label)
            train_set.append(data)

    label_set = [data[-1] for data in train_set]
    label_set = sorted(list(set(label_set)))
    for data in train_set:
        idx = label_set.index(data[-1])
        data[-1] = np.zeros(len(label_set))
        data[-1][idx] = 1
    cnt = len(train_set)
    return train_set[:int(cnt*0.9)], train_set[int(cnt*0.9):]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='iris.data file', required=True)
    args = parser.parse_args()

    return args


def run(args):
    net = Network([4, 3], learn_rate=0.03)
    net.evaluate(args.input)