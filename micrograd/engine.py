class Value:
    def __init__(self, data, prev=None):
        self.data = data
        if prev is not None:
            self._prev = prev
        else:
            self._prev = set()
        self.grad = 0.0

    def __add__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        d = self.data + b.data
        return Value(d, prev=set([self, b]))

    def __mul__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        d = self.data * b.data
        return Value(d, prev=set([self, b]))
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Value(data={self.data})"


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

    
        
    