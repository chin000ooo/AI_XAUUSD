"""Shared helpers for building the three notebooks with nbformat."""
from __future__ import annotations

import nbformat as nbf


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str):
    return nbf.v4.new_code_cell(source.strip("\n"))


def build(cells, path):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[built] {path}  ({len(cells)} cells)")
