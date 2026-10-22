"""Project paths and the chosen datasets. Everything stays inside this folder."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"            # genomic-benchmarks downloads (dest_path)
CACHE_DIR = ROOT / ".cache"         # k-mer matrices, sequence tables, mpl cache
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "mpl"))

# Chosen after inspecting list_datasets()/info(); see verification_report.md.
HUMAN = "human_enhancers_ensembl"
MOUSE = "dummy_mouse_enhancers_ensembl"
VERSION = 0
POSITIVE_CLASS = "positive"         # label 1 = regulatory (enhancer)

K_VALUES = [3, 4, 5, 6]
SEEDS = [0, 1, 2]
