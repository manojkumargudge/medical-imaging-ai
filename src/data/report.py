"""Generated Markdown files that hold several sections, each owned by one generator.

A section sits between `<!-- name:start -->` and `<!-- name:end -->`. Writing a section replaces
only that block, so docs/eda.md can hold the NIH and the spleen EDA without one generator deleting
the other's output.
"""

from __future__ import annotations

import os
from pathlib import Path


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- {name}:start -->", f"<!-- {name}:end -->"


def read_section(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    start, end = _markers(name)
    if start not in text:
        return None
    return text.split(start, 1)[1].split(end, 1)[0].strip("\n") + "\n"


def write_section(path: Path, name: str, body: str) -> None:
    """Replace the named block, or append it; text outside the block is kept as it is."""
    start, end = _markers(name)
    block = f"{start}\n{body.rstrip()}\n{end}\n"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if text.count(start) > 1 or text.count(start) != text.count(end):
        raise ValueError(f"{path}: malformed {name} markers")
    if start in text:
        before, rest = text.split(start, 1)
        after = rest.split(end, 1)[1].lstrip("\n")
        text = before + block + ("\n" + after if after else "")
    else:
        text = (text.rstrip("\n") + "\n\n" if text else "") + block
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8", newline="\n")
    os.replace(partial, path)
