# MSiSTFTの学習用のPyTorch Lightning Module

import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torchaudio
from pytorch_lightning import LightningModule
from pathlib import Path
from transformers import get_linear_schedule_with_warmup
import math
from src.tts.utils.audio import (
    slice_segments,
    spec_to_mel_torch,
    mel_spectrogram_torch
)
from src.config.config import Config

from src.models.msistftef2.generator_cn import Generator
from src.models.msistftef2.modules.hifigan import HiFiGANMultiScaleMultiPeriodDiscriminator
from src.models.msistftef2.modules.disc_loss import WavLMDiscriminator, WavLMLoss, generator_adversarial_loss, discriminator_adversarial_loss
from src.models.msistftef2.modules.utils import cosine_scheduler
from src.models.vits.loss import (
    GeneratorAdversarialLoss,
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    SISNRLoss,
    kl_divergence_loss_torch
)

from src.models.core.evaluator.manager import TTSEvaluateManager

from src.tts.utils.audio import save_wave
from src.utils.ml import get_dtype

class MsISTFTEF2Module(LightningModule):
    def __init__(
        self, 
        cfg: Config,
    ):
        super(MsISTFTEF2Module, self).__init__()
        self.cfg = cfg
        self.automatic_optimization = False
        # models
        self.net_g = Generator(cfg.model.net_g, cfg.dataset)
        self.net_d = HiFiGANMultiScaleMultiPeriodDiscriminator()
        # self.net_dur_d = DurationDiscriminator(
        #     cfg.model.net_g.z_channels,
        #     cfg.model.net_g.z_channels,
        #     kernel_size=cfg.model.net_g.duration_kernel_size
        # )
        
        self.wavlm_d = WavLMDiscriminator(
            slm_hidden=cfg.model.wavlm_d.hidden,
            slm_layers=cfg.model.wavlm_d.nlayers,
            initial_channel=cfg.model.wavlm_d.initial_channel
        )
        
        #loss
        loss_cfg = cfg.model.loss
        
        self.d_wav_lm_loss = WavLMLoss(
            model=cfg.model.wavlm_d.model,
            wd=self.wavlm_d,
            model_sr=cfg.dataset.sample_rate,
            slm_sr=cfg.model.wavlm_d.sr
        )

        self.feat_match_loss = FeatureMatchLoss(
            average_by_discriminators=loss_cfg.feat_match_loss.average_by_discriminators,
            average_by_layers=loss_cfg.feat_match_loss.average_by_layers,
            include_final_outputs=loss_cfg.feat_match_loss.include_final_outputs
        )
        
        if loss_cfg.sisnr_loss_use:
            self.si_snr_loss = SISNRLoss()
        
        #self._is_fp16 = cfg.ml.mix_precision == 16
        #self._scaler = GradScaler(enabled=self._is_fp16)
        self._dtype = get_dtype(cfg.ml.mix_precision)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = TTSEvaluateManager(
            sr=cfg.ml.evaluator.sr,
            speech_bert_score_model=cfg.ml.evaluator.speech_bert_score_model,
            device=self._device
        )
        self.val_output_dir = None
        self.val_resampler = torchaudio.transforms.Resample(cfg.dataset.sample_rate, cfg.ml.evaluator.sr)
        
        self.gamma_value_list = cosine_scheduler(
            loss_cfg.d_adv_loss.gamma_base,
            loss_cfg.d_adv_loss.gamma_final,
            cfg.ml.num_epochs,
        )
    
    def load_model_from_ckpt(
        self, net_g_path: str, net_d_path: str
    ):
        self.net_g.load_state_dict(torch.load(str(net_g_path)))
        self.net_d.load_state_dict(torch.load(str(net_d_path)))
    
    def generator_process(self, batch, batch_idx, step="train"):
        gamma_value = self.gamma_value_list[self.current_epoch]
        self.d_wav_lm_loss.wavlm.eval()
        optimizer_g, optimizer_d = self.optimizers()
        (
            wav_padded,
            _,
            spec_padded,
            spec_lengths,
            text_padded,
            text_lengths,
            accent_pos_padded,
            speaker_id,
        ) = batch
        
        with autocast(dtype=self._dtype):
            # Generator
            
            (
                wav_fake,
                (loss_a, loss_a_kl),
                attns,
                ids_slice,
                text_mask,
                z_mask,
                (m1, logs1, m_q1, logs_q1),
                (m2, logs2, m_q2, logs_q2),
                bi_kl,
                real_sigma
            ) = self.net_g(
                text_padded,
                text_lengths,
                accent_pos_padded,
                spec_padded,
                spec_lengths,
                speaker_id=speaker_id,
                bi=self.cfg.model.net_g.bidirectional
            )
            
            wav_real = slice_segments(
                    wav_padded,
                    ids_slice * self.cfg.dataset.hop_length,
                    self.cfg.dataset.segment_size,
                ) # id_slice * hop_lengthの位置からsegment_size分切り取る
           
            
            # 生成したwaveとOriginal spectrogramをMel Spectrogramに変換
            filter_length = self.cfg.dataset.filter_length
            sample_rate = self.cfg.dataset.sample_rate
            mel_bins = self.cfg.dataset.mel_bins
            segment_size = self.cfg.dataset.segment_size
            hop_length = self.cfg.dataset.hop_length
            win_length = self.cfg.dataset.win_length
            f_min = self.cfg.dataset.f_min
            f_max = self.cfg.dataset.f_max
            
            mel_spec_real = spec_to_mel_torch(
                spec_padded, filter_length, mel_bins, sample_rate, f_min, f_max
            )
            mel_spec_real = slice_segments(
                x=mel_spec_real,
                ids_str=ids_slice,
                segment_size=segment_size // hop_length,
            )
            
            mel_spec_fake = mel_spectrogram_torch(
                    wav_fake,
                    filter_length,
                    mel_bins,
                    sample_rate,
                    hop_length,
                    win_length,
                    fmin=f_min,
                    fmax=f_max
                )
            # lossを計算
        # Discriminatorの更新 (n_Dに一回更新)
        if (batch_idx + 1) % self.cfg.model.net_d.n_D_update_steps == 0:
            if step == "train":
                self.toggle_optimizer(optimizer_d)
            p_real, p_real_samples = self.net_d(wav_real, disc=True)
            p_fake, p_fake_samples = self.net_d(wav_fake.detach(), disc=True)
            
            # dur discriminator
            # y_dur_hat_r, y_hat_dur_g = self.net_dur_d(
            #     text_encoded.detach(),
            #     text_mask.detach(),
            #     logw_.detach(),
            #     logw.detach()
            # )
            # (
            #     adversarial_loss_dur_D, _, _ 
            # ) = discriminator_loss(y_dur_hat_r, y_hat_dur_g)
            
            # wavlm discriminator
            adversarial_loss_slm_D = 0 if step != "train" else self.d_wav_lm_loss.discriminator(
                wav_real.detach().squeeze(),
                wav_fake.detach().squeeze(),
                gamma=gamma_value
            ).mean()
            adversarial_loss_slm_D = (
                adversarial_loss_slm_D * self.cfg.model.loss.adversarial_loss_D_lambda
            )

            # lossを計算
            adversarial_loss_D = 0 if step != "train" else (
                discriminator_adversarial_loss(p_real, p_real_samples, p_fake, p_fake_samples, gamma=gamma_value)
                * self.cfg.model.loss.adversarial_loss_D_lambda
            )
            

        

            # log the loss
            self.log(
                f"{step}/adversarial_loss_D",
                adversarial_loss_D,
                on_step=True,
                prog_bar=True,
                logger=True,
            )
            
            self.log(
                f"{step}/adversarial_loss_wavlm_loss",
                adversarial_loss_slm_D,
                on_step=True,
                prog_bar=True,
                logger=True,
            )
            
            total_adversarial_loss_D = adversarial_loss_D + adversarial_loss_slm_D
            if step == "train":
                #self._scaler(adversarial_loss_D).backward()
                #self._scaler.unscale_(optimizer_d)
                self.manual_backward(total_adversarial_loss_D)
                
                if (batch_idx + 1) % int(self.cfg.ml.accumulate_grad_batches * self.cfg.model.net_d.n_D_update_steps) == 0:
                    self.clip_gradients(optimizer_d, gradient_clip_val=self.cfg.ml.grad_clip_val, gradient_clip_algorithm="norm")
                    optimizer_d.step()
                    optimizer_d.zero_grad()
                self.untoggle_optimizer(optimizer_d)
        
        if step == "train":
                self.toggle_optimizer(optimizer_g)
        
        # discriminatorの学習でnet_gは更新されていない(そのままの結果を使用)
        # Generatorノ更新
        p_real, _ = self.net_d(wav_real)
        p_fake, _ = self.net_d(wav_fake)

        
        # ef2 loss
        loss_a_total = self.cfg.model.loss.align_kl_loss_lambda * loss_a + loss_a_kl
        kl_loss3 = 0
        if self.cfg.model.net_g.bidirectional:
            m2, logs2, m_q2_r, logs_q2_r =  bi_kl
            kl_loss3= kl_divergence_loss_torch(m2, logs2, m_q2_r, logs_q2_r, z_mask) * self.cfg.model.loss.kl_loss_lambda
        
        
        # wavlm discriminator
        loss_wavlm_embs = self.d_wav_lm_loss(wav_real.detach().squeeze(), wav_fake.squeeze()).mean()
        loss_wavlm_adv_gen = self.d_wav_lm_loss.generator(wav_real.detach().squeeze(), wav_fake.squeeze())
        
        mel_reconstruction_loss = (
            F.l1_loss(mel_spec_real, mel_spec_fake) * self.cfg.model.loss.mel_loss_lambda
        )

        kl_loss1 = (
            kl_divergence_loss_torch(m1, logs1, m_q1, logs_q1, z_mask)
            * self.cfg.model.loss.kl_loss_lambda
        )
        
        kl_loss2 = (
            kl_divergence_loss_torch(m2, logs2, m_q2, logs_q2, z_mask)
            * self.cfg.model.loss.kl_loss_lambda
        )
        feature_matching_loss = (
            self.feat_match_loss(p_fake, p_real)
            * self.cfg.model.loss.feature_loss_loss_lambda
        )
        adversarial_loss_G = (
            generator_adversarial_loss(p_real, p_fake)
            * self.cfg.model.loss.adversarial_loss_G_lambda
        )
        
        adversarial_loss_wavlm_G = (
            loss_wavlm_adv_gen * self.cfg.model.loss.adversarial_loss_G_lambda
        )
        wavlm_embs_loss = (
            loss_wavlm_embs * self.cfg.model.loss.adversarial_loss_G_lambda
        )
        
        si_snr_loss = 0
        if self.cfg.model.loss.sisnr_loss_use:
            si_snr_loss = (
                self.si_snr_loss(wav_fake, wav_real)
                * self.cfg.model.loss.sisnr_loss_lambda
            )
            self.log(
                f"{step}/sisnr_loss",
                si_snr_loss,
                on_step=True,
                prog_bar=True,
                logger=True,
            )
            
        train_loss = (
                (kl_loss1 + kl_loss2 + kl_loss3 * 0.5) * 0.5
                + mel_reconstruction_loss
                + loss_a_total
                + feature_matching_loss
                + adversarial_loss_G
                + adversarial_loss_wavlm_G
                + wavlm_embs_loss
                + si_snr_loss
        )
        # log the loss
        self.log(
            f"{step}/kl_loss", (kl_loss1 + kl_loss2 + kl_loss3 * 0.5) * 0.5, on_step=True, prog_bar=True, logger=True
        )
        self.log(
            f"{step}/kl_loss1", kl_loss1, on_step=True, prog_bar=True, logger=True
        )
        self.log(
            f"{step}/kl_loss2", kl_loss2, on_step=True, prog_bar=True, logger=True
        )
        self.log(
            f"{step}/kl_loss_bi", kl_loss3, on_step=True, prog_bar=True, logger=True
        )
        self.log(
            f"{step}/mel_reconstruction_loss",
            mel_reconstruction_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step}/loss_a_total",
            loss_a_total,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step}/loss_align",
            loss_a,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step}/loss_align_kl",
            loss_a_kl,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step}/feature_matching_loss",
            feature_matching_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step}/adversarial_loss_G",
            adversarial_loss_G,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step}/adversarial_loss_wavlm_G",
            adversarial_loss_wavlm_G,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step}/wavlm_embs_loss",
            wavlm_embs_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step}/total_loss",
            train_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        
    
        
        if step == "train":
            self.manual_backward(train_loss)
            if (batch_idx + 1) % self.cfg.ml.accumulate_grad_batches == 0:
                self.clip_gradients(optimizer_g, gradient_clip_val=self.cfg.ml.grad_clip_val, gradient_clip_algorithm="norm")
                optimizer_g.step()
                optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)
        
        
            
    def training_step(self, batch, batch_idx, ):        
        self.generator_process(batch, batch_idx, step="train")
        if self.cfg.model.scheduler_g.use and self.cfg.model.scheduler_d.use:
            sh_g, sh_d = self.lr_schedulers()
            sh_g.step()
            sh_d.step()
        
        self.net_g.update_current_mas_noise_scale()
        self.log("mas_noise_scale", self.net_g.current_mas_noise_scale, on_epoch=True, prog_bar=True, logger=True)
        
    def validation_step(self, batch, batch_idx):
        # Loss計算・Log作成
        self.generator_process(batch, batch_idx, step="val")

        # Batchの一部でSample VCを行う
        (
            real_wave_padded,
            wav_lengths,
            _,
            _,
            text_padded,
            text_lengths,
            accent_pos_padded,
            speaker_id,
        ) = batch
        
        batch_size = real_wave_padded.size(0)
        
        for i in range(batch_size):
            with autocast(dtype=self._dtype):
                fake_wave, attns, y_mask = self.net_g.text_to_speech(
                    text_padded[i:i+1, :text_lengths[i]],
                    text_lengths[i:i+1],
                    accent_pos_padded[i:i+1, :text_lengths[i]],
                    speaker_id[i:i+1]
                ) # (1, 1, len) => (1, len)
            y_hat_lengths = y_mask.sum([1, 2]).long() * self.cfg.dataset.hop_length
            fake_wave = fake_wave[:, :, :y_hat_lengths[0]]
            fake_wave = fake_wave[0]
            if fake_wave.size()[-1] < self.cfg.dataset.sample_rate * 1:
                fake_wave = F.pad(fake_wave, (0, self.cfg.dataset.sample_rate * 1 - fake_wave.size()[-1]))
            # 評価
            real_wave_length = wav_lengths[i]
            real_wave = real_wave_padded[i, :, :real_wave_length] # 1バッチ目のみ
            
            self.evaluator.evaluate(self.val_resampler(real_wave), self.val_resampler(fake_wave))
            
            # 結果を保存
            fake_wave = fake_wave.detach().cpu()
            self.val_output_dir = Path(self.cfg.path.val_out_dir) / f"{self.current_epoch:05d}"
            self.val_output_dir.mkdir(parents=True, exist_ok=True)

            wave_index = int(batch_idx * batch_size + i)
            if wave_index % self.cfg.ml.wav_save_every_n == 0:
                output_path = self.val_output_dir / f"gen_{wave_index:05d}.wav"
                
                save_wave(
                    fake_wave,
                    str(output_path),
                    sample_rate=self.cfg.dataset.sample_rate,
                )
        return None
    
    def on_validation_epoch_start(self) -> None:
        self.evaluator.reset()
        return None
    def on_validation_epoch_end(self) -> None:
        result = self.evaluator.get_results()
        self.log("val/xvector_sim", result.xvector_sim, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/gen_mos", result.gen_mos, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/gen_mos_rate", result.gen_mos_rate, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/speech_bert_score_precision", result.speech_bert_score_precision, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/speech_bert_score_recall", result.speech_bert_score_recall, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/speech_bert_score_f1", result.speech_bert_score_f1, on_epoch=True, prog_bar=True, logger=True)
        with open(self.val_output_dir / "result.txt", "w") as f:
            f.write(f"xvector_sim: {result.xvector_sim}\n")
            f.write(f"gen_mos: {result.gen_mos}\n")
            f.write(f"gen_mos_rate: {result.gen_mos_rate}\n")
            f.write(f"speech_bert_score_precision: {result.speech_bert_score_precision}\n")
            f.write(f"speech_bert_score_recall: {result.speech_bert_score_recall}\n")
            f.write(f"speech_bert_score_f1: {result.speech_bert_score_f1}\n")
        
        return None

    def configure_optimizers(self):
        optg_cfg = self.cfg.model.optim_g
        optd_cfg = self.cfg.model.optim_d
        shg_cfg = self.cfg.model.scheduler_g
        shd_cfg = self.cfg.model.scheduler_d
        
        if optg_cfg.name == "AdamW":
            optimizer_g = torch.optim.AdamW(
                self.net_g.parameters(),
                lr=optg_cfg.lr,
                eps=optg_cfg.eps,
                betas=optg_cfg.betas,
            )
        else:
            raise ValueError(f"Optimizer {optg_cfg.name} is not supported.")

        if optd_cfg.name == "AdamW":
            optimizer_d = torch.optim.AdamW(
                list(self.net_d.parameters()) + list(self.d_wav_lm_loss.parameters()),
                lr=optd_cfg.lr,
                eps=optd_cfg.eps,
                betas=optd_cfg.betas,
            )
        else:
            raise ValueError(f"Optimizer {optd_cfg.name} is not supported.")
        
        training_step = math.ceil(self.cfg.dataset.train_dataset_num / self.cfg.ml.batch_size / self.cfg.ml.accumulate_grad_batches) * self.cfg.ml.num_epochs
        if self.cfg.model.scheduler_g.use and self.cfg.model.scheduler_d.use:
            if shg_cfg.name == "ExponentialLR":
                scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer_g, gamma=shg_cfg.gamma
                )
            elif shg_cfg.name == "linear_w_warmup":
                scheduler_g = get_linear_schedule_with_warmup(
                    optimizer=optimizer_g,
                    num_warmup_steps=int(self.cfg.dataset.train_dataset_num / self.cfg.ml.batch_size / self.cfg.ml.accumulate_grad_batches * shg_cfg.warmup_epoch),
                    num_training_steps=training_step,
                )
            else:
                raise ValueError(f"Scheduler {shg_cfg.name} is not supported.")
            
            if shd_cfg.name == "ExponentialLR":
                scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer_d, gamma=shd_cfg.gamma
                )
            elif shd_cfg.name == "linear_w_warmup":
                scheduler_d = get_linear_schedule_with_warmup(
                    optimizer=optimizer_d,
                    num_warmup_steps=int(self.cfg.dataset.train_dataset_num / self.cfg.ml.batch_size / self.cfg.ml.accumulate_grad_batches * shd_cfg.warmup_epoch),
                    num_training_steps=training_step,
                )
                
            else:
                raise ValueError(f"Scheduler {shd_cfg.name} is not supported.")
            
            #scheduler_g = {
            #    "scheduler": scheduler_g,
            #    "interval": shg_cfg.interval,
            #}
            #scheduler_d = {
            #    "scheduler": scheduler_d,
            #    "interval": shd_cfg.interval,
            #}
            return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
        return [optimizer_g, optimizer_d], []