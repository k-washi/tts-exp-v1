from dataclasses import dataclass, field

@dataclass()
class SaveConfig:
    top_k: int = -1
    monitor: str = 'val/total_loss'
    mode: str = "min"

@dataclass()
class EvalatorConfig:
    sr: int = 16000
    speech_bert_score_model: str = "japanese-hubert-base"

@dataclass()
class MLConfig:
    seed:int =  3407
    batch_size: int = 32
    num_workers: int = 4
    accumulate_grad_batches: int = 1
    grad_clip_val: float = 0.5
    num_epochs: int = 100
    check_val_every_n_epoch: int = 5
    max_steps: int = 500000
    mix_precision: str = 32 # 16 or 32, bf16
    gpu_devices: int = 1
    profiler: str = "simple"
    checkpoint: SaveConfig = field(default_factory=lambda: SaveConfig())
    evaluator: EvalatorConfig = field(default_factory=lambda: EvalatorConfig())
    wav_save_every_n: int = 10