import torch
import numpy as np
from transformers import AutoModel
from typing import Union

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def bert_score(v_generated, v_reference):
    """
    Args:
        v_generated (torch.Tensor): Generated feature tensor (T, D).
        v_reference (torch.Tensor): Reference feature tensor (T, D).
    Returns:
        float: Precision.
        float: Recall.
        float: F1 score.
    """
    # Calculate cosine similarity
    sim_matrix = torch.matmul(v_generated, v_reference.T) / (torch.norm(v_generated, dim=1, keepdim=True) * torch.norm(v_reference, dim=1).unsqueeze(0))

    # Calculate precision and recall
    precision = torch.max(sim_matrix, dim=1)[0].mean().item()
    recall = torch.max(sim_matrix, dim=0)[0].mean().item()

    # Calculate F1 score
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score

class SpeechBertScore():
    def __init__(
        self,
        sr=16000,
        model_type="japanese-hubert-base",
        use_gpu=True
    ) -> None:
        if model_type == "japanese-hubert-base":
            model_name = "rinna/japanese-hubert-base"
            self.model = AutoModel.from_pretrained(model_name)
        else:
            raise ValueError(f"model_type {model_type} not supported")
        
        self.model.eval()
        
        # Use GPU if available
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        
        self.sr = sr
        assert sr == 16000, "Only 16kHz sampling rate is supported"
        
    
    def process_feats(self, audio:torch.Tensor):
        with torch.no_grad():
            outputs = self.model(input_values=audio)
            return outputs.last_hidden_state # (1, seq_len, hidden_size)
    
    
        
    def score(
        self, 
        gt_wav:Union[np.ndarray, torch.Tensor], 
        gen_wav:Union[np.ndarray, torch.Tensor]
    ):
        """
        Args:
            gt_wav (np.ndarray or torch.Tensor): Ground truth waveform (T,).
            gen_wav (np.ndarray or torch.Tensor): Generated waveform (T,).
        Returns:
            float: Precision.
            float: Recall.
            float: F1 score.
        """
        if isinstance(gt_wav, np.ndarray):
            gt_wav = torch.from_numpy(gt_wav)
        if isinstance(gen_wav, np.ndarray):
            gen_wav = torch.from_numpy(gen_wav)
        
        if gt_wav.dim() == 1:
            gt_wav = gt_wav.unsqueeze(0)
        if gen_wav.dim() == 1:
            gen_wav = gen_wav.unsqueeze(0)
            
        gt_wav = gt_wav.to(self.device).float()
        gen_wav = gen_wav.to(self.device).float()
        
        v_reference = self.process_feats(gt_wav)
        v_generated = self.process_feats(gen_wav)
        precition, recall, f1_score = bert_score(v_generated.squeeze(0), v_reference.squeeze(0))
        return precition, recall, f1_score


if __name__ == "__main__":
    import torchaudio
    test_wav = "src/__example/sample.wav"
    audio, sr = torchaudio.load(test_wav)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    audio = resampler(audio).squeeze(0)
    speech_bert_score = SpeechBertScore(
        model_type="japanese-hubert-base",
        use_gpu= True if torch.cuda.is_available() else False
    )
    
    precition, recall, f1_score = speech_bert_score.score(audio, audio)
    print(precition, recall, f1_score)
    
    
    