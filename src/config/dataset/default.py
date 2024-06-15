from dataclasses import dataclass, field

@dataclass()
class DatasetConfig:
    # データ関連
    n_speaker: int = 1
    sample_rate: int = 22050
    filter_length: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    mel_bins: int = 80
    segment_size: int = 16384
    f_max: int = 11025
    f_min: int = 0
    add_blank_type: int = 0 # 0: なし, 1: 音素の前後に挿入, 2: モーラの前後に挿入
    accent_split: bool = False # アクセントを分割するか
    accent_up_ignore: bool = False # アクセント上昇を無視するか
    use_distirubute_sampler: bool = True
    train_dataset_num: int = 4500