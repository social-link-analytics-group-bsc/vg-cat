import os
import yaml
import logging
from pathlib import Path

def setup_logging(log_path=None, level=logging.INFO):
    logging.basicConfig(
        filename=log_path,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}