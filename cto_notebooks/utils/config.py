import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    cache_dir: Path
    model_dir: Path


def _init_config() -> Config:
    cache_dir = Path.resolve(Path(__file__ + "/../../../cache/"))
    model_dir = Path.resolve(Path(__file__ + "/../../../models/"))
    cache_hub_dir = cache_dir.joinpath("hub")
    model_dir.mkdir(exist_ok=True)
    cache_hub_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = cache_dir.as_posix()
    os.environ["HF_HUB_CACHE"] = cache_hub_dir.as_posix()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return Config(cache_dir=cache_dir, model_dir=model_dir)


CONFIG: Config = _init_config()
