"""wav, phonome+accentのデータセットモジュール"""
import torch
import traceback
from torch.utils.data import Dataset
from pathlib import Path

from src.config.config import Config
from src.tts.utils.audio import (
    load_wave,
    spectrogram_torch
)

from src.tts.phonome.utils import (
    read_symbols
)

from src.tts.phonome.param import (
    symbol_preprocess,
    phonome_to_sequence,
    accent_to_sequence
)

class AudioTextDataset(Dataset):
    def __init__(self, cfg: Config, data_list: list):
        """
        Initializes an instance of the AudioTextDataset class.

        Args:
            cfg (Config): The configuration object.
            data_list (list): A list of data names.

        """
        self.cfg = cfg
        self.data_list = data_list
        
        
        self._wave_dir = Path(cfg.path.dataset_dir) / "wav"
        self._phoneme_dir = Path(cfg.path.dataset_dir) / "phoneme"
        
        # 複数人は未対応
        assert cfg.dataset.n_speaker == 1, f"{cfg.dataset.n_speaker} speakers are not supported"
        self._sid = 0
        self.dist_sampler_setup()
    
    def dist_sampler_setup(self):
        """
        Sets up the distribution sampler by calculating the lengths of the waveforms.

        """
        lengths = []
        for name in self.data_list:
            audio_path = self._wave_dir / f"{name}.wav"
            waveform, _ = load_wave(str(audio_path), sample_rate=self.cfg.dataset.sample_rate, is_torch=True, mono=True)
            length = len(waveform) / self.cfg.dataset.sample_rate
            lengths.append(length)
        self.lengths = lengths
        
        
    
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: A tuple containing the waveform, spec, phoneme_indexes, accent_indexes, and sid.

        """
        name = self.data_list[idx]
        
        audio_path = self._wave_dir / f"{name}.wav"
        waveform, spec = self.get_audio(audio_path)
        text_path = self._phoneme_dir / f"{name}.txt"
        phonome_indexes, accent_indexes = self.get_phonomes(text_path)
        phonome_indexes = torch.LongTensor(phonome_indexes)
        accent_indexes = torch.LongTensor(accent_indexes)
        return waveform, spec, phonome_indexes, accent_indexes, self._sid
        
        
    def get_audio(self, audio_path: Path):
        """
        Loads the audio waveform and computes the spectrogram.

        Args:
            audio_path (Path): The path to the audio file.

        Returns:
            tuple: A tuple containing the waveform and the spectrogram.

        """
        waveform, _ = load_wave(str(audio_path), sample_rate=self.cfg.dataset.sample_rate, is_torch=True, mono=False)
        # waveform = waveform / torch.abs(waveform).max() * 0.9
        spec = spectrogram_torch(
            waveform, 
            self.cfg.dataset.filter_length,
            self.cfg.dataset.sample_rate,
            self.cfg.dataset.hop_length,
            self.cfg.dataset.win_length
        )[0]
        return waveform, spec
    
    def get_phonomes(self, text_path: Path):
        """
        Reads the phoneme symbols from the text file and converts them to indexes.

        Args:
            text_path (Path): The path to the text file.

        Returns:
            tuple: A tuple containing the phoneme indexes and the accent indexes.

        """
        try:
            symbol_list = read_symbols(str(text_path))
            phonome_list, accent_list = symbol_preprocess(
                symbol_list, 
                add_blank_type=self.cfg.dataset.add_blank_type,
                accent_split=self.cfg.dataset.accent_split,
                accent_up_ignore=self.cfg.dataset.accent_up_ignore
            )
            phonome_indexes = phonome_to_sequence(phonome_list)
            accent_indexes = accent_to_sequence(accent_list)
        except Exception as e:
            print(traceback.format_exc())
            raise ValueError(f"Error occurred at {text_path}")
        return phonome_indexes, accent_indexes
        
        
        