"""Execute a notebook with project root as cwd.

Usage:
    python run_notebook.py [notebook_path]

Default: Multiasset Notebooks/Extended_Dataset_Figures.ipynb
"""
import asyncio
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).parent

if len(sys.argv) > 1:
    NB_PATH = PROJECT_ROOT / sys.argv[1]
else:
    NB_PATH = PROJECT_ROOT / "Multiasset Notebooks" / "Extended_Dataset_Figures.ipynb"

stem = NB_PATH.stem
OUT_PATH = NB_PATH.parent / f"{stem}_executed.ipynb"

# Change to project root so relative paths work
os.chdir(PROJECT_ROOT)

nb = nbformat.read(str(NB_PATH), as_version=4)

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(PROJECT_ROOT)}},
)

print(f"Executing notebook from: {PROJECT_ROOT}")
print(f"Notebook: {NB_PATH.name}")

client.execute()

nbformat.write(nb, str(OUT_PATH))
print(f"\nDone! Executed notebook saved to: {OUT_PATH}")
