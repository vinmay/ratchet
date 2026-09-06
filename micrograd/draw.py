"""Printing helpers for the expression graph.

Not part of the engine. Reads `Value.data` and `Value._prev`, nothing else.

    python -m micrograd.draw
"""


def format_graph(node):
    """Return a tree drawing of the graph that produced `node`.

    The root is the result; everything below it is what fed it. A node reached
    more than once (the diamond case) is drawn once and then marked, so the
    sharing is visible instead of silently duplicated.
    """
    lines = []
    seen = set()

    def walk(n, prefix, connector):
        repeated = n in seen
        seen.add(n)
        mark = "   <- same node as above" if repeated else ""
        lines.append(f"{prefix}{connector}{n!r}{mark}")

        if repeated or not n._prev:
            return

        # _prev is a set and sets have no order; sort so runs are reproducible
        parents = sorted(n._prev, key=lambda p: p.data)

        if connector == "":
            child_prefix = prefix
        elif connector.startswith("└"):
            child_prefix = prefix + "   "
        else:
            child_prefix = prefix + "│  "

        for i, parent in enumerate(parents):
            last = i == len(parents) - 1
            walk(parent, child_prefix, "└─ " if last else "├─ ")

    walk(node, "", "")
    return "\n".join(lines)


def print_graph(node):
    print(format_graph(node))


if __name__ == "__main__":
    from micrograd.engine import Value

    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    print("a * b + c")
    print_graph(a * b + c)

    print()
    x = Value(3.0)
    print("x * x + x")
    print_graph(x * x + x)
