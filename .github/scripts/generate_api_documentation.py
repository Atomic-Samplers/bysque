"""Generate API reference pages and render `zensical.toml` from
`_zensical.toml`.

Run before `zensical build` / `zensical serve`:

    uv run python docs/generate_api_documentation.py

Writes one stub per module under `docs/reference/`, then replaces the
`"@api-reference@"` token in `_zensical.toml` with the generated nav section
and writes the result to `zensical.toml`.

Both `docs/reference/` and `zensical.toml` are generated: gitignore them.
"""

from __future__ import annotations

import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

SRC = Path("src")
REFERENCE = Path("docs/reference")
TEMPLATE = Path("_zensical.toml")
OUTPUT = Path("zensical.toml")

ANCHOR = '{ "Code Documentation" = "reference/" }'

SECTION = "API reference"
IGNORE = {"__init__", "__main__", "_version", "_cli"}

Tree = dict[str, Any]


def collect() -> Tree:
    """Build a nested {name: subtree | doc path} mapping from `src/`."""
    tree: Tree = {}
    for path in sorted(SRC.rglob("*.py")):
        parts = path.relative_to(SRC).with_suffix("").parts
        if any(part in IGNORE for part in parts):
            continue

        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = "/".join(parts) + ".md"
    return tree


def write_stubs(tree: Tree, prefix: tuple[str, ...] = ()) -> None:
    """Write one mkdocstrings stub per module, titled to avoid auto-
    capitalization.
    """
    for name, value in tree.items():
        if isinstance(value, dict):
            write_stubs(value, (*prefix, name))
            continue

        target = REFERENCE / value
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"---\ntitle: {name}\n---\n\n::: {'.'.join((*prefix, name))}\n")


def render(tree: Tree) -> str:
    """Render the tree as a single-line TOML array (inline tables cannot span
    lines).
    """
    entries = [
        f"{{ '{name}' = {render(value)
        if isinstance(value, dict) else f"'{REFERENCE.name}/{value}'"} }}"
        for name, value in sorted(tree.items(), key=lambda kv: (isinstance(kv[1], dict), kv[0]))
    ]
    return "[" + ", ".join(entries) + "]"


def main() -> None:
    tree = collect()
    if not tree:
        sys.exit(f"no modules found under {SRC}/")

    template = TEMPLATE.read_text()
    if ANCHOR not in template:
        sys.exit(f"anchor not found in {TEMPLATE}:\n  {ANCHOR}")

    shutil.rmtree(REFERENCE, ignore_errors=True)
    write_stubs(tree)

    section = ANCHOR.split(" = ")[0] + f" = {render(tree)} }}"
    OUTPUT.write_text(template.replace(ANCHOR, section))

    with OUTPUT.open("rb") as fd:
        tomllib.load(fd)


if __name__ == "__main__":
    main()
