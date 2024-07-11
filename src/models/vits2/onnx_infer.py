import numpy as np
import onnxruntime
from src.config.config import Config
from src.tts.phonome.utils import OpenJtalk
from src.tts.phonome.param import (
    symbol_preprocess,
    phonome_to_sequence,
    accent_to_sequence
)
from src.tts.utils.normalize_text import normalize_text

class VitsInferenceModel():
    def __init__(
        self,
        model_path,
        sr=22050,
        device="cpu",
        user_dic=None
    ) -> None:
        if device == "cpu" or device is None:
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise RuntimeError("Unsportted Device")
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        self.sampling_rate = sr
        
        self.openjtalk = OpenJtalk(user_dic=user_dic)
        
    
    def forward(self, phonome_indexes, text_lengths, accent_indexes, sid):
        for i in self.model.get_inputs():
            print(i.name)
        onnx_inputs = {
            self.model.get_inputs()[0].name: phonome_indexes,
            #self.model.get_inputs()[1].name: text_lengths,
            self.model.get_inputs()[1].name: accent_indexes,
            self.model.get_inputs()[2].name: sid
        }
        return self.model.run(None, onnx_inputs)
        
    
    def inference(self, x: str, kana=False, phoneme=False):
        if kana:
            x = normalize_text(x)
            symbol_list = self.openjtalk.kana2symbols(x)
        elif phoneme:
            symbol_list = x.split("-")
        else:
            x = normalize_text(x)
            symbol_list = self.openjtalk.extract_symbols(x)
        print("Input:", x)
        print("Symbol:", symbol_list)
        phonome_list, accent_list = symbol_preprocess(symbol_list, add_blank=True)
        phonome_indexes = phonome_to_sequence(phonome_list)
        accent_indexes = accent_to_sequence(accent_list)
        
        phonome_indexes = np.array(phonome_indexes, dtype=np.int64).reshape(1, -1)
        accent_indexes = np.array(accent_indexes, dtype=np.int64).reshape(1, -1)
        sid = np.array([0], dtype=np.int64)
        text_lengths = np.array([len(phonome_indexes[0])], dtype=np.int64)
        print(phonome_indexes.shape, accent_indexes.shape, sid.shape, text_lengths.shape)
        
        return self.forward(phonome_indexes, text_lengths, accent_indexes, sid)[0][0]

if __name__ == "__main__":
    import argparse
    from src.tts.utils.audio import save_wave
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/vits_best_model.onnx")
    parser.add_argument("--text", type=str, default="お客様の課題解決にあたって、新たなテクノロジーを活用した各種システム開発から各種コンサルティングまで、様々なサービスを幅広く提供しています。")
    parser.add_argument("--output", type=str, default="./data/output.wav")
    parser.add_argument("--kana", action="store_true")
    parser.add_argument("--phoneme", action="store_true")
    parser.add_argument("--user_dic", type=str, default="./data/user_dic.dic")
    parser.add_argument("--sr", type=int, default=22050)
    
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    text = args.text
    output = args.output
    sr = args.sr

    
    vits = VitsInferenceModel(
        checkpoint_path,
        sr=sr,
        user_dic=args.user_dic
    )
    
    o = vits.inference(text, kana=args.kana, phoneme=args.phoneme)
    print(o.shape)
    
    save_wave(o, output, sample_rate=sr)