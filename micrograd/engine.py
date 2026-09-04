class Value:
    def __init__(self, data, prev=None):
        self.data = data
        if prev is not None:
            self._prev = prev
        else:
            self._prev = set()

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



    