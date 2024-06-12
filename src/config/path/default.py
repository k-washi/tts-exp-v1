from dataclasses import dataclass, field
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent.parent)

@dataclass()
class PathConfig:
    root_dir: str = root_dir
    model_save_dir: str = "checkpoints"
    dataset_dir: str = "data/jsut"
    val_out_dir: str = "val"
    train_file_path: str = "vits/dataset/jsut/train.txt"
    valid_file_path: str = "vits/dataset/jsut/val.txt"
    