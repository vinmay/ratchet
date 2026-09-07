import random
from micrograd.engine import Value

class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(1, -1)) for _ in range(nin)]
        self.bias = Value(0.1)

    def __call__(self, inputs):
        weighted_sum = sum(i * weight for i, weight in zip(inputs, self.weights))
        activation = weighted_sum + self.bias
        return activation.tanh()

    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = []
        for i in range(nout):
            self.neurons.append(Neuron(nin))

    def __call__(self, inputs):
        activation = []
        for neuron in self.neurons:
            activation.append(neuron(inputs))
        return activation

    def parameters(self):
        return [value for neuron in self.neurons for value in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts # We do this because layer 0s result goes into layer 1 and so on
        self.layers = []
        for i in range(len(sizes)-1):
            self.layers.append(Layer(sizes[i], sizes[i+1]))

    def __call__(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer(out)
        if len(out) == 1: 
            return out[0] 
        else: 
            return out

    def parameters(self):
        return [value for layer in self.layers for value in layer.parameters()]

    def zero_grad(self):
        params = self.parameters()
        for param in params:
            param.grad = 0.0

def mse_loss(pred, actual):
    squared_errors = [(Value(act) + (-1 * pred)) ** 2 for act, pred in zip(actual, pred) ]
    mse = sum(squared_errors) * (1/len(pred))
    return mse
        