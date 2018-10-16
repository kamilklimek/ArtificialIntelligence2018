import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Layer:
    def __init__(self, input, height):
        self.weights = np.random.rand(height, input)
        self.biases = np.random.rand(height)

    def feed_forward(self, input):
        o = np.dot(self.weights, input.T) + self.biases
        o = [sigmoid(x) for x in o]
        return o

    def __str__(self):
        return "Weights: " + str(self.weights) + "\n Biases: " + str(self.biases)


class NeuralNetwork:
    def __init__(self, archetype):
        self.layers = self.create_layers_by_archetype(archetype)
        self.archetype = archetype

    def __init__(self, name_file):
        self.loadNetworkFromFile(name_file)

    def create_layers_by_archetype(self, archetype):
        layers = []
        for i in range(0, len(archetype)-1):
            layer = Layer(archetype[i], archetype[i+1])
            layers.append(layer)
        return layers

    def feed_forward(self, inputs):
        outputs = inputs.copy()
        for layer in self.layers:
            outputs = np.array(layer.feed_forward(outputs))
        return outputs

    def saveNetworkToFile(self, file_name):
        file_handler = open(file_name, "w")
        file_handler.write(str(self.archetype) + "\n")
        for layer in self.layers:
            file_handler.write(str(layer.weights) + "\n")
            file_handler.write(str(layer.biases) + "\n")

        file_handler.close()

    def loadNetworkFromFile(self, file_name):
        file_handler = open(file_name, "r")
        self.archetype = np.array(file_handler.readline())
        for i in range (0, len(self.archetype)-1):
            layer = Layer()
            layer.biases = []
            layer.weights = []


nn = NeuralNetwork([6, 5, 4, 4, 5])
outputs = nn.feed_forward(np.array([1, 1, 1, 1, 1, 1]))
nn.saveNetworkToFile("percepton_1.txt")
nn.loadNetworkFromFile("percepton_1.txt")
print(outputs)
