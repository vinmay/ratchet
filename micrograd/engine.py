from math import exp as math_exp
from math import tanh as math_tanh
class Value:
    def __init__(self, data, prev=None):
        self.data = data
        if prev is not None:
            self._prev = prev
        else:
            self._prev = set()
        self.grad = 0.0
        self._backward = lambda: None

    def __add__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        d = self.data + b.data
        result = Value(d, prev=set([self, b]))

        def _backward():
            self.grad += 1 * result.grad 
            b.grad += 1 * result.grad
        result._backward = _backward
        return result

    def __mul__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        d = self.data * b.data
        result = Value(d, prev=set([self, b]))

        def _backward():
            self.grad += b.data * result.grad 
            b.grad += self.data * result.grad
                
        result._backward = _backward
        return result

    def __pow__(self, other):
        d = self.data ** other
        result = Value(d, prev=set([self]))
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * result.grad    
        result._backward = _backward
        return result

    def exp(self):
        d = math_exp(self.data)
        result = Value(d, prev=set([self]))
        def _backward():
            self.grad += result.data * result.grad
        result._backward = _backward
        return result

    def tanh(self):
        d = math_tanh(self.data)
        result = Value(d, prev=set([self]))
        def _backward():
            self.grad += (1 - d ** 2) * result.grad
        result._backward = _backward
        return result

    def relu(self):
        d = self.data if self.data > 0 else 0.0
        result = Value(d, prev=set([self]))
        def _backward():
            self.grad += result.grad if self.data > 0 else 0.0
        result._backward = _backward
        return result
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Value(data={self.data})"

    def backward(self):
        self.grad = 1.0
        order = topo_sort(self)
        for i in order[::-1]:
            i._backward()


def topo_sort(node):

    def recursive(node):
        if node not in visited:
            visited.add(node)
            for i in node._prev:
                recursive(i)
            order.append(node)


    visited = set()
    order = []
    recursive(node)
    return order

def grad_check(f, x, h=1e-5):
    result = (f(x + h) - f(x-h))/(2*h)
    return result
