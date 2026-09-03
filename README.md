# ratchet

Building a language model from nothing. No PyTorch, no TensorFlow, no autograd
library. A scalar autograd engine, then a tensor engine, then a transformer,
then a trained model that generates text.

The name is the method. A ratchet only turns one way. Progress here is
append-only: every unit ends with a passing test, and the project never
restarts from the beginning.

## What this is

A learning project, in the open. The goal is not a competitive model. The goal
is that every abstraction between `loss.backward()` and the arithmetic
underneath it is one I have written myself and can defend.

Everything is built on Python and NumPy. Anything that would ordinarily be
imported is instead implemented:

| Normally you `import` | Here it is built in |
|---|---|
| `torch.Tensor` + autograd | `tensorgrad/tensor.py` |
| `nn.Linear`, `nn.LayerNorm` | `tensorgrad/layers.py` |
| `optim.SGD`, `optim.Adam` | `tensorgrad/optim.py` |
| `F.scaled_dot_product_attention` | `gpt/attention.py` |
| `F.cross_entropy` | fused softmax + CE, by hand |
| A tokenizer | `gpt/data.py` |

## Why build it this way

I have spent ten years on distributed systems, and I am comfortable reasoning
about things whose internals I understand. Machine learning has mostly been an
API surface to me: call the framework, get a number, trust it. That is a bad
place to reason from.

The specific thing I want is the ability to look at a training run that is
going wrong and know why, rather than searching for someone who has seen the
same symptom.

## How I am doing it

Code first, mathematics on demand. The usual approach to this material front
loads weeks of derivatives and linear algebra before any code runs, which gives
no feedback signal and no clear point at which you are prepared enough to
begin. This project inverts that ordering.

### The gradient checker is the teacher

Every backward pass is written from rough intuition and then verified
numerically:

```python
def grad_check(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
```

If the analytical gradient matches the numerical one, the reasoning was right.
If it does not, there is a specific, findable error. Math becomes a debugging
tool used at the moment it is needed, rather than a prerequisite to be cleared
in advance.

The practical effect: it is impossible to be "not ready to start." There is
always a next line of code.

### The math surface is finite

The full set of mathematics required to get from an empty file to a trained
transformer:

1. Chain rule, as products of local derivatives along a path
2. Derivatives of eight primitives: `+`, `*`, `pow`, `exp`, `log`, `tanh`/`relu`, `max`, `sum`
3. Matmul backward: `dA = dC @ B.T`, `dB = A.T @ dC`, derivable from shape-matching
4. Softmax composed with cross-entropy collapses to `(p - y)`
5. Broadcasting forward means summing over the broadcast dimensions backward

Each is encountered at the point where it blocks a line of code, and nowhere
else. Treated as an open-ended subject, the mathematics is intimidating. Scoped
to what the code actually requires, it is five items.

### Twelve units, each ending in something that runs

Not a reading list. Every unit produces working code and at least one test.

| # | Unit | Output |
|---|---|---|
| 1 | Expression graph | `Value` class, topological sort |
| 2 | Backward + gradient checker | Every primitive gradchecked |
| 3 | Train an MLP | Loss descends on a 2D toy dataset |
| 4 | Break it deliberately | Catalogued failure signatures |
| 5 | Tensors | Matmul and broadcast backward |
| 6 | Layers and real data | MNIST above 90%, own engine |
| 7 | Optimizers | SGD, momentum, Adam compared |
| 8 | Bigram language model | Tokenizer, batching, sampling |
| 9 | Attention | Single head, gradchecked, beats bigram |
| 10 | The full block | Multi-head, LayerNorm, residuals |
| 11 | Training | Real run on an RTX 4060, readable output |
| 12 | Inference | Sampling strategies, KV cache |

Unit 4 is not padding. Deliberately reproducing exploding losses, dead
gradients, missing `zero_grad`, and zero-initialized weights builds the pattern
recognition that Unit 11 depends on entirely.

### Constraints I am holding to

- **No coding agents through Unit 10.** If an agent writes the autograd engine,
  the exercise is void. From Unit 11 they may handle data loading, checkpoint
  IO, and plotting. Never the model.
- **No copying reference implementations.** Reading them after solving
  something is fine and often useful. Reading them before is not.
- **Every backward pass gets a test.** `tests/` grows with every unit and is
  run at the start of every session.
- **Sessions end on a commit and one line in `LOG.md`.** If the build is broken
  when time runs out, it gets stashed and the symptom written down.
- **No restarts.** After any gap, work resumes at the last passing test rather
  than at Unit 1.

`CLAUDE.md` defines what a coding agent is permitted to do in this repo. The
short version: not the model, not the gradients, not the engine. It has been in
the repository since the first commit.

## Progress

| Unit | Status |
|---|---|
| 1. Expression graph | in progress |
| 2. Backward + checker | |
| 3. Train an MLP | |
| 4. Break it | |
| 5. Tensors | |
| 6. Layers + MNIST | |
| 7. Optimizers | |
| 8. Bigram LM | |
| 9. Attention | |
| 10. Full block | |
| 11. Train | |
| 12. Inference + ship | |

`LOG.md` has the session-by-session detail.

## Layout

```
ratchet/
  SYLLABUS.md          # the full plan, with definitions of done
  CLAUDE.md            # what a coding agent may and may not write here
  LOG.md               # one line per session
  micrograd/           # Units 1-4: scalar autograd
    engine.py
    nn.py
  tensorgrad/          # Units 5-7: array autograd
    tensor.py
    layers.py
    optim.py
  gpt/                 # Units 8-12: the model
    data.py
    attention.py
    model.py
    train.py
    sample.py
  tests/
    test_gradcheck.py
```

## Running it

```bash
git clone <repo>
cd ratchet
pip install numpy
python -m pytest tests/
```

Nothing else is required. That is somewhat the point.

## Hardware

Development and everything through Unit 10 runs on CPU. Unit 11 trains on a
single RTX 4060 with 8GB of VRAM. No cloud compute is used anywhere in this
project, which constrains model size and is a deliberate part of the exercise.

## References

Read for direction, not for code:

- Andrej Karpathy, *Neural Networks: Zero to Hero*
- Vaswani et al., *Attention Is All You Need*
- Ba et al., *Layer Normalization*
- Kingma and Ba, *Adam*

## License

MIT. See [`LICENSE`](LICENSE).
