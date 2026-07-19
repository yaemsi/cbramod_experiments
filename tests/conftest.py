from __future__ import annotations

import torch


def pytest_sessionstart() -> None:
    """Keep tiny CPU unit tests fast and deterministic on large CI machines."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
