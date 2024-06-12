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
    # 各音素の間に<BLK>を追加。blank_itemが「0」の場合、<PAD>を追加
    blank_version: int = 0
    use_distirubute_sampler: bool = True