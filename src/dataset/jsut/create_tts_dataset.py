# jsutデータセット+ラベルから、TTSモデル学習用の形式に変更
from pathlib import Path
import shutil
import librosa as lr
import numpy as np

import soundfile as sf
import torchaudio

from tqdm import tqdm
from src.tts.frontend.openjtalk import pp_symbols
from src.tts.frontend.hts import HTSLabelFile
from src.tts.phonome.utils import save_symbols
from src.tts.utils.audio import load_wave

def load_np_wav(fp, sr):
    w, _sr = lr.load(fp, sr=sr, mono=True)
    if w.dtype in [np.int16, np.int32]:
        w = (w / np.iinfo(w.dtype).max).astype(np.float64)
    return w

def save_np_wav(fp, data, sr):
    sf.write(fp, data, sr, subtype='PCM_24')

def create_tts_dataset_from_jsut(
    jsut_dir, 
    label_dir, 
    output_dir, 
    sr,
    start_del_time,
    end_add_time,
    partial_frame
    
):
    wav_dir = Path(jsut_dir) / "basic5000"/ "wav"
    wav_list = list(wav_dir.glob("*.wav"))
    assert len(wav_list) > 0, f"{wav_dir}は空です"
    
    lab_dir = Path(label_dir) / "labels" / "basic5000"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_wav_dir = output_dir / "wav"
    output_wav_dir.mkdir(exist_ok=True)
    output_txt_dir = output_dir / "phonome"
    output_txt_dir.mkdir(exist_ok=True)
    
    hts = HTSLabelFile()
    
    min_length = 1000
    max_length = 0
    for wav_fp in tqdm(wav_list):
        fn = wav_fp.stem
        lab_fp = lab_dir / f"{fn}.lab"
        
        # labfile to phonome
        labels = hts.load(str(lab_fp))
        PP = pp_symbols(labels.contexts)
        save_symbols(PP, str(output_txt_dir / f"{fn}.txt"))
        
        # enhance
        w, _ = load_wave(str(wav_fp), sample_rate=sr, is_torch=True, mono=False)
        w = w.detach().numpy().copy()[0]
        
        # audio
        #w = load_np_wav(str(wav_fp), sr=sr)
        
        # wavをpartial_frameの倍数に正規化
        # min_frame~max_frameの抜き出す
        min_frame = round((labels.start_times[1]/ 10000000 - start_del_time)*sr)
        min_frame = min_frame - min_frame % partial_frame
        w_min_frame = max(0, min_frame)

        max_frame = round((labels.end_times[-2]/ 10000000 + end_add_time)*sr)
        max_frame = max_frame - max_frame % partial_frame + partial_frame

        w_len = len(w) - len(w) % partial_frame
        w_max_frame = min(max_frame, w_len)
        
        trim_w = w[w_min_frame: w_max_frame]
        min_length = min(min_length, len(trim_w)/sr)
        max_length = max(max_length, len(trim_w)/sr)
        save_np_wav(str(output_wav_dir / f"{fn}.wav"), trim_w, sr)
    print(f"min_length: {min_length}, max_length: {max_length}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j", "--jsut_dir", default="data/jsut_ver1.1"
    )
    parser.add_argument(
        "-l", "--label_dir", default="data/jsut-label"
    )
    parser.add_argument(
        "-o", "--output_dir", default="data/jsut"
    )
    parser.add_argument(
        "-sr", "--sr", default=22050, type=int
    )
    parser.add_argument(
        "-st", "--start_del_time", default=0.05, type=float
    )
    parser.add_argument(
        "-et", "--end_add_time", default=0.1, type=float
    )
    parser.add_argument(
        "-pf", "--partial_frame", default=256, type=int, help="このフレーム数の倍数にする"
    )
    
    
    args = parser.parse_args()
    create_tts_dataset_from_jsut(
        args.jsut_dir,
        args.label_dir,
        args.output_dir,
        args.sr,
        args.start_del_time,
        args.end_add_time,
        args.partial_frame
    )