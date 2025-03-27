import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from transformers import AutoModel
from typing import Union, List, Tuple



def zero_centerd_gradient_penality(
    samples: torch.Tensor,
    critics: List[torch.nn.Module],
):
    gradient, = torch.autograd.grad(outputs=critics.sum(), inputs=samples, create_graph=True)
    return gradient.square().sum([1, 2])

def generator_adversarial_loss(
    real_outputs,
    fake_outputs,
):
    if isinstance(real_outputs, (tuple, list)):
        loss = 0.0
        for i, (real_outputs_, fake_outputs_) in enumerate(zip(real_outputs, fake_outputs)):
            if isinstance(real_outputs_, (tuple, list)):
                real_outputs_ = real_outputs_[-1]
                fake_outputs_ = fake_outputs_[-1]
            if real_outputs_.dim() == 3:
                real_outputs_ = real_outputs_.squeeze(1)
            if fake_outputs_.dim() == 3:
                fake_outputs_ = fake_outputs_.squeeze(1)
            relative_logits = fake_outputs_ - real_outputs_.detach()
            
            adv_loss = F.softplus(-relative_logits)
            adv_loss = adv_loss.sum(dim=-1) / adv_loss.size(-1)
            loss += adv_loss
        loss /= (i + 1)
    else:
        relative_logits = fake_outputs - real_outputs.detach()
        adv_loss = F.softplus(-relative_logits) / relative_logits.size(-1)
        loss = adv_loss
    return loss.mean()

def discriminator_adversarial_loss(
    real_outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    real_samples: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    fake_outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    fake_samples: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    gamma: float = 1.0,
):
    if isinstance(real_outputs, (tuple, list)):
        loss = 0.0
        for i, (
            real_outputs_, real_samples_, fake_outputs_, fake_samples_
        ) in enumerate(zip(real_outputs, real_samples, fake_outputs, fake_samples)):
            if isinstance(real_outputs_, (tuple, list)):
                real_outputs_ = real_outputs_[-1]
                fake_outputs_ = fake_outputs_[-1]
            if real_outputs_.dim() == 2:
                real_outputs_ = real_outputs_.unsqueeze(1)
            if fake_outputs_.dim() == 2:
                fake_outputs_ = fake_outputs_.unsqueeze(1)
            if real_samples_.dim() == 2:
                real_samples_ = real_samples_.unsqueeze(1)
            if fake_samples_.dim() == 2:
                fake_samples_ = fake_samples_.unsqueeze(1)
            # real_samples_ = real_samples_.detach().requires_grad_(True)
            # fake_samples_ = fake_samples_.detach().requires_grad_(True)
            r1_penalty = zero_centerd_gradient_penality(real_samples_, real_outputs_)
            r2_penalty = zero_centerd_gradient_penality(fake_samples_, fake_outputs_)
            relative_logits = real_outputs_ - fake_outputs_
            adv_loss = F.softplus(-relative_logits).sum(dim=-1) / relative_logits.size(-1)
            loss += adv_loss + (gamma / 2) * (r1_penalty + r2_penalty)
        loss /= (i + 1)
    else:
        # real_samples = real_samples.detach().requires_grad_(True)
        # fake_samples = fake_samples.detach().requires_grad_(True)
        r1_penalty = zero_centerd_gradient_penality(real_samples, real_outputs)
        r2_penalty = zero_centerd_gradient_penality(fake_samples, fake_outputs)
        relative_logits = real_outputs - fake_outputs
        adv_loss = F.softplus(-relative_logits).sum(dim=-1) / relative_logits.size(-1)
        loss = adv_loss + (gamma / 2) * (r1_penalty + r2_penalty)
    print(loss.mean())
    return loss.mean()

class WavLMDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(
        self, slm_hidden=768, slm_layers=13, initial_channel=64, use_spectral_norm=False
    ):
        super(WavLMDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.pre = norm_f(
            nn.Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0)
        )

        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv1d(
                        initial_channel, initial_channel * 2, kernel_size=5, padding=2
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        initial_channel * 2,
                        initial_channel * 4,
                        kernel_size=5,
                        padding=2,
                    )
                ),
                norm_f(
                    nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)
                ),
            ]
        )

        self.conv_post = norm_f(nn.Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def forward(self, x):
        x = self.pre(x)

        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x

class WavLMLoss(torch.nn.Module):
    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        #self.wavlm.eval()
        #for param in self.wavlm.parameters():
        #    param.requires_grad = False

    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16.squeeze(), output_hidden_states=True
        ).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            wav_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
                
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16.squeeze(), output_hidden_states=True
        ).hidden_states
        
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        r_logits = self.wd(wav_embeddings.detach())
        y_df_hat_g = self.wd(y_rec_embeddings)
        relative_logits = y_df_hat_g - r_logits
        adv_loss = F.softplus(-relative_logits)
        loss_gen = adv_loss.mean()

        return loss_gen

    def discriminator(self, wav, y_rec, gamma=0.05):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(
                input_values=y_rec_16, output_hidden_states=True
            ).hidden_states

            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_embeddings = y_embeddings.detach().requires_grad_(True)
        y_rec_embeddings = y_rec_embeddings.detach().requires_grad_(True)
        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)
        r1_penalty = zero_centerd_gradient_penality(y_embeddings, y_d_rs)
        r2_penalty = zero_centerd_gradient_penality(y_rec_embeddings, y_d_gs)

        relative_logits = y_d_rs - y_d_gs
        adv_loss = F.softplus(-relative_logits).sum(dim=-1) / relative_logits.size(-1)
        
        adv_loss = adv_loss + (gamma / 2) * (r1_penalty + r2_penalty)
        return adv_loss.mean()

    def discriminator_forward(self, wav):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs


if __name__ == "__main__":
    pass