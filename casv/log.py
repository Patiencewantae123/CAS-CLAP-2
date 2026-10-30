"""Logging setup shared by all entry points."""

from __future__ import annotations

import logging
import warnings


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure timestamped console logging and quieten third-party libraries."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("transformers", "diffusers", "huggingface_hub", "urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    return logging.getLogger("casv")
