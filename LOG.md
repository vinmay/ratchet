# Log

One line per session. Written in the last three minutes, before the commit.

Format:

```
YYYY-MM-DD  U<unit>  <what works now>. Next: <the single next thing>.
```

Rules:

- **"Next" is mandatory and must be one concrete action.** This is the whole
  point of the file. It removes the decision cost at the start of the next
  session. "Next: keep working on tensors" is a wasted line. "Next: unbroadcast
  for the case where a dim was size 1" is a line that starts a session.
- Write it even on a bad session. Especially on a bad session.
- If the build is broken at the end, stash it and record the symptom.
- Do not edit old lines. Append only.

After a gap of any length, you open this file, read the last line, and do that
thing. You do not reread the syllabus and you do not restart.

---

2026-09-02  U1  Repo scaffolded. Docs, deps, test harness green. Next: `Value.__init__` with data, grad, _prev.

2026-09-03  U1  `Value` with `data` and `_prev`. `__add__`/`__mul__` coerce int/float operands and record parents; `__radd__`/`__rmul__` delegate. Leaves get an empty set. Tests 1-3 green. Next: `topo_sort(node)` as a module-level function — DFS over `_prev`, append each node after its children, skip nodes already in a visited set.

2026-09-05  U1  Unit 1 done bar one line. `__repr__`, and `topo_sort` as a module-level function — recursive DFS over `_prev`, visited set, append after children. All 8 tests green including the diamond. `micrograd/draw.py` prints the graph as a tree and marks shared nodes. Next: `self.grad = 0.0` in `Value.__init__` — the last Unit 1 checklist item — then Unit 2 opens with a `_backward` closure on `__add__`.

2026-09-06  U2  Unit 2 done. `_backward` closures on `+`, `*`, `__pow__`, `exp`, `tanh`, `relu`, each attached to the node the operation creates and accumulating with `+=`. `backward()` seeds the root at 1.0 and walks `topo_sort` in reverse. `grad_check` uses central differences at h=1e-5. 93 tests green; every primitive is checked against the numerical gradient to 1e-6. Note: `backward()` does not zero grads, so Unit 3's loop needs an explicit `zero_grad`. Next: `Neuron` in `micrograd/nn.py` — weights and bias as `Value`s, `__call__` doing the weighted sum plus activation.

2026-09-07  U3  Unit 3 done — the network learns. `Neuron`, `Layer`, `MLP` in `micrograd/nn.py`, all built on the six primitives from Unit 2; `MLP.__call__` threads the input through the layers and unwraps only when the last layer has one neuron. `mse_loss` stays a `Value` end to end — taking `.data` inside it severs the graph and silently gives 0/37 gradients. `train.py` at the repo root: seeded, 100 epochs, lr=0.05 on 100 moons. Loss 1.21 -> 0.36, accuracy 87%, and the ASCII boundary is curved. 138 tests green. Next: Unit 4, reproduce a dead `tanh` by initialising every weight to the same large value and record what the loss curve looks like.
