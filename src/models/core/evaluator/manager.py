import torch
import torchaudio
from dataclasses import dataclass
from src.models.core.evaluator.evaluator import TTSEvaluator

@dataclass
class TTSEvaluateResult():
    xvector_sim:float
    gen_mos:float
    gen_mos_rate:float
    speech_bert_score_precision:float
    speech_bert_score_recall:float
    speech_bert_score_f1:float

class TTSEvaluateManager():
    def __init__(
        self,
        sr=16000,
        speech_bert_score_model:str = "japanese-hubert-base",
        device:str = "cpu",
    ):
        self.evaluator = TTSEvaluator(
            sr=sr,
            speech_bert_score_model=speech_bert_score_model,
            device=device
        )
        self.sr = sr
    
    def resample(self, audio:torch.Tensor, raw_sr):
        if raw_sr != self.sr:
            resampler = torchaudio.transforms.Resample(raw_sr, self.sr)
        return resampler(audio)
        
    def reset(self):
        self.xvector_sim_list = []
        self.gen_mos_list = []
        self.gen_mos_rate_list = []
        self.speech_bert_score_precision_list = []
        self.speech_bert_score_recall_list = []
        self.speech_bert_score_f1_list = []
        
    def evaluate(
        self,
        ref_audio:torch.Tensor,
        gen_audio:torch.Tensor,
    ):
        xvector_sim, (gen_mos, gen_mos_rate), (precision, recall, f1) = self.evaluator.process(
            ref_audio=ref_audio,
            gen_audio=gen_audio
        )
        
        self.xvector_sim_list.append(xvector_sim)
        self.gen_mos_list.append(gen_mos)
        self.gen_mos_rate_list.append(gen_mos_rate)
        self.speech_bert_score_precision_list.append(precision)
        self.speech_bert_score_recall_list.append(recall)
        self.speech_bert_score_f1_list.append(f1)
    
    def get_results(self):
        assert len(self.xvector_sim_list) == len(self.gen_mos_list) \
            == len(self.gen_mos_rate_list) == len(self.speech_bert_score_precision_list) \
            == len(self.speech_bert_score_recall_list) == len(self.speech_bert_score_f1_list), \
            "The number of evaluation results do not match"
            
        assert len(self.xvector_sim_list) > 0, "No evaluation results found"
        return TTSEvaluateResult(
            xvector_sim=sum(self.xvector_sim_list) / len(self.xvector_sim_list),
            gen_mos=sum(self.gen_mos_list) / len(self.gen_mos_list),
            gen_mos_rate=sum(self.gen_mos_rate_list) / len(self.gen_mos_rate_list),
            speech_bert_score_precision=sum(self.speech_bert_score_precision_list) / len(self.speech_bert_score_precision_list),
            speech_bert_score_recall=sum(self.speech_bert_score_recall_list) / len(self.speech_bert_score_recall_list),
            speech_bert_score_f1=sum(self.speech_bert_score_f1_list) / len(self.speech_bert_score_f1_list)
        )
        
