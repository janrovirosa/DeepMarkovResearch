"""Execute Extended_Dataset_Figures.ipynb with project root as cwd."""
import asyncio
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).parent
NB_PATH = PROJECT_ROOT / "Multiasset Notebooks" / "Extended_Dataset_Figures.ipynb"
OUT_PATH = PROJECT_ROOT / "Multiasset Notebooks" / "Extended_Dataset_Figures_executed.ipynb"

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
