"""Stop before an experiment can exhaust the dataset/checkpoint filesystem."""
from pathlib import Path
import os
import shutil


def require_space(path, min_free_gib=8):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free / 1024**3
    if free < min_free_gib:
        raise RuntimeError(f"STOP: {path} has {free:.2f} GiB free; requires {min_free_gib:.2f} GiB. No further experiment steps.")
    return free


def atomic_torch_save(payload, path):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()
