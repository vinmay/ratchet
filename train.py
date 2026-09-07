from micrograd.nn import MLP, mse_loss
from micrograd.data import make_moons, ascii_scatter
import random

random.seed(0)
N = 100
lr = 0.05

X, y = make_moons(n=100, noise=0.1, seed=0)

nin = len(X[0])
model = MLP(nin, [4, 4, 1])
params = model.parameters()

for epoch in range(N):
    model.zero_grad()
    predictions = []
    for x in X:
        predictions.append(model(x))
    loss = mse_loss(predictions, y)
    loss.backward()
    for param in params:
        param.data -= lr * param.grad
    print(f'Epoch:{epoch} loss:{loss.data:.4f}')

acc = sum((model(x).data > 0) == (t > 0) for x, t in zip(X, y)) / len(X)
print(f"accuracy {acc:.0%}")

ascii_scatter(X, y, predict=lambda p:model(p).data)