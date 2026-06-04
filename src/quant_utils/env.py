"""
env.py — environment detection, path bootstrap, reproducibility.

Works identically on Colab and local. No machine-specific hardcoded paths.
"""
from __future__ import annotations

import os
import sys
import random
import platform
import subprocess
from pathlib import Path


def detect_env() -> dict:
    """Return a small dict describing the runtime environment."""
    is_colab = ("google.colab" in sys.modules) or (os.environ.get("COLAB_RELEASE_TAG") is not None)
    info = {
        "is_colab": is_colab,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
    }
    return info


def find_project_root(start: str | None = None,
                      markers=("requirements.txt", "src", ".git", "notebooks")) -> Path:
    """Walk upward from ``start`` until a directory containing a marker is found."""
    p = Path(start or os.getcwd()).resolve()
    for cand in [p, *p.parents]:
        if any((cand / m).exists() for m in markers):
            return cand
    return p


def setup_project_root(explicit: str | None = None) -> Path:
    """Resolve the project root and put ``<root>/src`` on sys.path so
    ``import quant_utils`` works from a notebook in ``<root>/notebooks``."""
    root = Path(explicit).resolve() if explicit else find_project_root()
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def ensure_dirs(root: str | Path) -> None:
    """Create the standard output folders if missing (idempotent)."""
    root = Path(root)
    for d in ["data/raw", "data/external", "data/processed", "models", "reports"]:
        (root / d).mkdir(parents=True, exist_ok=True)


def mount_drive(mountpoint: str = "/content/drive") -> bool:
    """Colab-only Google Drive mount. No-op (with warning) elsewhere."""
    try:
        from google.colab import drive  # type: ignore
        drive.mount(mountpoint)
        return True
    except Exception as e:  # pragma: no cover - colab only
        print(f"[WARNING] Drive mount skipped (not on Colab or failed): {e}")
        return False


def set_seeds(seed: int = 42) -> int:
    """Seed every RNG we can reach and log it. Returns the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    for mod in ("torch",):  # optional frameworks
        try:
            m = __import__(mod)
            m.manual_seed(seed)  # type: ignore
        except Exception:
            pass
    print(f"[SEED] all RNGs seeded with {seed} (PYTHONHASHSEED={seed})")
    return seed


def log_gpu() -> str | None:
    """Best-effort GPU detection (Colab/CUDA). Never raises."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            print(f"[GPU] {out.stdout.strip()}")
            return out.stdout.strip()
    except Exception:
        pass
    print("[GPU] none detected — running on CPU")
    return None


def banner(env: dict, root: Path) -> None:
    """Print a compact environment banner for the notebook header."""
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"  colab        : {env['is_colab']}")
    print(f"  platform     : {env['platform']}")
    print(f"  python       : {env['python']}")
    print(f"  project root : {root}")
    print("=" * 60)
