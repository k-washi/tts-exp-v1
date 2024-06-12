from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from typing import List, Any


from src.config.dataset.default import DatasetConfig
from src.config.path.default import PathConfig
from src.config.ml.default import MLConfig

from src.config.model.vits import ModelConfig

defaults = [
    {"dataset": "default"},
    {"ml": "default"},
    {"path": "default"},
    {"model": "vits"}
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    dataset: DatasetConfig= MISSING
    ml: MLConfig = MISSING
    path: PathConfig = MISSING
    model: ModelConfig = MISSING


def get_config(
    dataset: DatasetConfig = DatasetConfig(),
    ml: MLConfig = MLConfig(),
    path: PathConfig = PathConfig(),
    model: ModelConfig = ModelConfig()
):
    cfg = OmegaConf.create({
        "_target_": "__main__.Config",
        "dataset": dataset,
        "ml": ml,
        "path": path,
        "model": model
    })
    cfg = hydra.utils.instantiate(cfg)
    return cfg