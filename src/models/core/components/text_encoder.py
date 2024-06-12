import math
from typing import Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass

import src.models.core.components.net_utils as net_utils
from src.models.core.components.conformer.encoder import Encoder

@dataclass
class EncoderCFG:
    attention_heads:int = 2 # transformerに似た構造のモジュールで使われている、MultiHeadAttentionのhead数
    ffn_expand:int = 4
    blocks:int = 6
    positionwise_layer_type:str = "conv1d"
    positionwise_conv_kernel_size:int = 3
    positional_encoding_layer_type: str = "rel_pos"
    self_attention_layer_type: str = "rel_selfattn"
    activation_type: str = "swish"
    normalize_before: bool = True
    use_macaron_style: bool = False
    use_conformer_conv: bool = False
    conformer_kernel_size: int = 7
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size, # 音素の種類数
        accent_position_size=0,  # アクセントの種類数
        input_embedding_dim=192, # 音素の埋め込みベクトルDim
        enccfg: EncoderCFG = EncoderCFG()
        
    ) -> None:
        super().__init__()
        self.__input_embedding_dim = input_embedding_dim
        
        self.emb = torch.nn.Embedding(vocab_size, input_embedding_dim)
        torch.nn.init.normal_(self.emb.weight, 0.0, input_embedding_dim ** -0.5)
        
        if accent_position_size > 0:
            self.accent_emb = torch.nn.Embedding(accent_position_size, input_embedding_dim)
            nn.init.normal_(self.accent_emb.weight, 0.0, input_embedding_dim ** -0.5)
            self.accent_proj = torch.nn.Conv1d(
                input_embedding_dim, input_embedding_dim, 1
            )
        else:
            self.accent_emb = None
            self.accent_proj = None
            
        self.encoder = Encoder(
            attention_dim=input_embedding_dim,
            attention_heads=enccfg.attention_heads,
            linear_units=int(enccfg.ffn_expand * input_embedding_dim),
            num_blocks=enccfg.blocks,
            dropout_rate=enccfg.dropout_rate,
            positional_dropout_rate=enccfg.positional_dropout_rate,
            attention_dropout_rate=enccfg.attention_dropout_rate,
            normalize_before=enccfg.normalize_before,
            positionwise_layer_type=enccfg.positionwise_layer_type,
            positionwise_conv_kernel_size=enccfg.positionwise_conv_kernel_size,
            macaron_style=enccfg.use_macaron_style,
            pos_enc_layer_type=enccfg.positional_encoding_layer_type,
            selfattention_layer_type=enccfg.self_attention_layer_type,
            activation_type=enccfg.activation_type,
            use_cnn_module=enccfg.use_conformer_conv,
        )
        self.proj = torch.nn.Conv1d(input_embedding_dim, input_embedding_dim * 2, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        accent_pos_padded=None
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.
        Args:
            x (Tensor): Input index tensor (B, T_text).
            x_lengths (Tensor): Length tensor (B,).
        Returns:
            Tensor: Encoded hidden representation (B, attention_dim, T_text).
            Tensor: Projected mean tensor (B, attention_dim, T_text).
            Tensor: Projected scale tensor (B, attention_dim, T_text).
            Tensor: Mask tensor for input tensor (B, 1, T_text).
        """
        
        # アクセント
        if self.accent_emb is not None:
            accent_emb = self.accent_emb(accent_pos_padded) # (B, text_length, input_embedding_dim)
            accent_emb = self.accent_proj(accent_emb.transpose(1, 2)).transpose(1, 2)
            x = (self.emb(x) + accent_emb) * math.sqrt(self.__input_embedding_dim)
        else:
            x = self.emb(x) * math.sqrt(self.__input_embedding_dim)
     
        x_mask = (
            net_utils.make_non_pad_mask(x_lengths)
            .to(
                device=x.device,
                dtype=x.dtype
            )
        ).unsqueeze(1)
        x, _ = self.encoder(x, x_mask)

        # convert the channel first (B, attention_dim, T_text)
        x = x.transpose(1, 2)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(stats.size(1) // 2, dim=1)
        return x, m, logs, x_mask
    

if __name__ == "__main__":
    t = TextEncoder(
        5,
        input_embedding_dim=4,
        accent_position_size=2,
        enccfg=EncoderCFG()
    )
    
    t_lengths = [10, 5]
    text = torch.ones((2, 10), dtype=torch.long)
    accent =  torch.ones((2, 10), dtype=torch.long)
    x, m, logs, x_mask = t(text, torch.tensor(t_lengths), accent)
    print(x.shape, m.shape, logs.shape, x_mask.shape)