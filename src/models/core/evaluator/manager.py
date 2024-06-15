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
        
        if xvector_sim is not None:
            self.xvector_sim_list.append(xvector_sim)
        if gen_mos is not None:
            self.gen_mos_list.append(gen_mos)
        if gen_mos_rate is not None:
            self.gen_mos_rate_list.append(gen_mos_rate)
        if precision is not None:
            self.speech_bert_score_precision_list.append(precision)
        if recall is not None:
            self.speech_bert_score_recall_list.append(recall)
        if f1 is not None:
            self.speech_bert_score_f1_list.append(f1)
    
    def get_results(self):
        xvector_sim = sum(self.xvector_sim_list) / len(self.xvector_sim_list) if len(self.xvector_sim_list) > 0 else 0
        gen_mos = sum(self.gen_mos_list) / len(self.gen_mos_list) if len(self.gen_mos_list) > 0 else 0
        gen_mos_rate = sum(self.gen_mos_rate_list) / len(self.gen_mos_rate_list) if len(self.gen_mos_rate_list) > 0 else 0
        speech_bert_score_precision = sum(self.speech_bert_score_precision_list) / len(self.speech_bert_score_precision_list) if len(self.speech_bert_score_precision_list) > 0 else 0
        speech_bert_score_recall = sum(self.speech_bert_score_recall_list) / len(self.speech_bert_score_recall_list) if len(self.speech_bert_score_recall_list) > 0 else 0
        speech_bert_score_f1 = sum(self.speech_bert_score_f1_list) / len(self.speech_bert_score_f1_list) if len(self.speech_bert_score_f1_list) > 0 else 0
        return TTSEvaluateResult(
            xvector_sim=xvector_sim,
            gen_mos=gen_mos,
            gen_mos_rate=gen_mos_rate,
            speech_bert_score_precision=speech_bert_score_precision,
            speech_bert_score_recall=speech_bert_score_recall,
            speech_bert_score_f1=speech_bert_score_f1
        )

        
