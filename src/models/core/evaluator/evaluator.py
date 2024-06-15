import torch
import torchaudio
import torch.nn.functional as F
from torchaudio.compliance import kaldi
from src.models.core.evaluator.speechbertscore import SpeechBertScore

import traceback

class TTSEvaluator:
    def __init__(
        self,
        sr=16000,
        speech_bert_score_model:str = "japanese-hubert-base",
        device:str = "cpu",
    ):
        assert sr == 16000, "Only 16kHz sampling rate is supported"
        self.sr = sr
        self.xvector = torch.hub.load("sarulab-speech/xvector_jtubespeech", "xvector", trust_repo=True)
        self.xvector.eval()
        
        self.speech_mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        self.speech_mos.eval()
        
        self.speech_bert_score = SpeechBertScore(
            sr=sr,
            model_type=speech_bert_score_model,
            use_gpu= True if torch.cuda.is_available() else False
        )
        
        self.device = device
        self.xvector.to(self.device)
        self.speech_mos.to(self.device)
    
    def process(
        self,
        ref_audio:torch.Tensor,
        gen_audio:torch.Tensor,
    ):
        """
        Process the reference audio and generated audio to compute various evaluation metrics.

        Args:
            ref_audio (torch.Tensor): The reference audio waveform.
            gen_audio (torch.Tensor): The generated audio waveform.

        Returns:
            tuple: A tuple containing the following evaluation metrics:
                - xvector_score (float): The cosine similarity score between the reference and generated x-vectors.
                - speech_mos_metrics (tuple): A tuple containing the following speech MOS metrics:
                    - gen_speech_mos (float): The mean opinion score (MOS) of the generated speech.
                    - speech_mos_rate (float): The MOS rate of the generated speech relative to the reference speech.
                - bert_score (float): speech bert mos score (precision, recall, f1).
        """
        if ref_audio.dim() == 1:
            ref_audio = ref_audio.unsqueeze(0)
        if gen_audio.dim() == 1:
            gen_audio = gen_audio.unsqueeze(0)
        ref_audio = ref_audio.to(self.device).float()
        gen_audio = gen_audio.to(self.device).float()
        
        ref_mfcc = kaldi.mfcc(ref_audio, num_ceps=24, num_mel_bins=24).unsqueeze(0)
        gen_mfcc = kaldi.mfcc(gen_audio, num_ceps=24, num_mel_bins=24).unsqueeze(0)
        with torch.no_grad():
            try:
                ref_xvector = self.xvector.vectorize(ref_mfcc)
                gen_xvector = self.xvector.vectorize(gen_mfcc)
                xvector_score = F.cosine_similarity(ref_xvector, gen_xvector)
                xvector_score = xvector_score.item()
            except Exception as e:
                print(f"Xvecter Error: {traceback.format_exc()}")
                xvector_score = None
            try:
                ref_speech_mos = self.speech_mos(ref_audio, self.sr)
                gen_speech_mos = self.speech_mos(gen_audio, self.sr)
                speech_mos_rate =  gen_speech_mos / ref_speech_mos
                gen_speech_mos = gen_speech_mos.item()
                speech_mos_rate = speech_mos_rate.item()
            except Exception as e:
                print(f"Speech MOS Error: {traceback.format_exc()}")
                gen_speech_mos = None
                speech_mos_rate = None
                
            try:
                bert_score = self.speech_bert_score.score(ref_audio, gen_audio)
            except Exception as e:
                print(f"Speech Bert Score Error: {traceback.format_exc()}")
                bert_score = (None, None, None)
        
        return xvector_score,  (gen_speech_mos, speech_mos_rate),  bert_score

if __name__ == "__main__":
    evaluator = TTSEvaluator()
    ref_audio, sr = torchaudio.load("src/__example/sample.wav")
    resampler = torchaudio.transforms.Resample(sr, 16000)
    ref_audio = resampler(ref_audio)
    gen_audio = ref_audio
    xvector_score, speech_mos, bert_score = evaluator.process(ref_audio, gen_audio)
    print(xvector_score, speech_mos, bert_score)
        