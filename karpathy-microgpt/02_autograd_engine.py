# $ python3 02_autograd_engine.py
#
# Goal: the `Value` class -- a scalar-valued autograd engine, the same
# design as Karpathy's earlier `micrograd` project, condensed to what
# microgpt.py needs. Every number that should be learnable gets wrapped in
# a `Value`; arithmetic on `Value`s builds a computation graph as a side
# effect, and `.backward()` walks that graph in reverse topological order,
# applying the chain rule at each node to fill in every `.grad`. This one
# class is the entire "how does the model learn" mechanism -- steps 3-5
# just build bigger expressions out of it.
# Step 2: Value (scalar autograd) -- forward ops build the graph, backward() fills gradients

import math                        # math.log, math.exp
import random
random.seed(42)

# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')  # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads  # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# --- everything below is this step's own demo, not part of the original file ---
# f(a, b) = a*b + b**3, by hand: df/da = b, df/db = a + 3*b**2
a, b = Value(-2.0), Value(3.0)
f = a * b + b**3
f.backward()
print(f"f(a,b) = a*b + b**3 = {f.data}")
print(f"df/da = {a.grad}  (expected {b.data})")
print(f"df/db = {b.grad}  (expected {a.data + 3 * b.data**2})")

# check df/da against a finite-difference approximation, as a sanity check
eps = 1e-6
a2 = Value(-2.0 + eps)
f2 = a2 * b + b**3
numerical = (f2.data - f.data) / eps
print(f"finite-difference df/da ~= {numerical:.6f}  (autograd said {a.grad})")
