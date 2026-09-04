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
