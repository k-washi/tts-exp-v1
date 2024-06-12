import torch
import torch.nn as nn
import src.models.core.components.net_utils as net_utils

from src.models.core.components.conformer.convolution import ConvolutionModule
from src.models.core.components.conformer.encoder_layer import EncoderLayer
import src.models.core.components.transformer.embedding as pos_embedding
import src.models.core.components.transformer.attention as attention
from src.models.core.components.transformer.layer_norm import LayerNorm
from src.models.core.components.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from src.models.core.components.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward
)
from src.models.core.components.transformer.repeat import repeat


class Encoder(nn.Module):
    # https://github.com/espnet/espnet/blob/95b582730f7a6eb04920951fb2320685d507c8a0/espnet/nets/pytorch_backend/conformer/encoder.py
    def __init__(
        self,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        pos_enc_layer_type="abs_pos",
        selfattention_layer_type="selfattn",
        activation_type="swish",
        use_cnn_module=False,
        zero_triu=False,
        cnn_module_kernel=31,
        stochastic_depth_rate=0.0,
    ) -> None:
        super().__init__()
        activation = net_utils.get_activation(activation_type)
        # Position Embedding
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = pos_embedding.PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = pos_embedding.RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
        
        self.embed = torch.nn.Sequential(
            pos_enc_class(attention_dim, positional_dropout_rate)
        )
        
        self.normalize_before = normalize_before
        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = attention.MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos", f"rel attention should pos_enc to rel_pos, but {pos_enc_layer_type}"
            encoder_selfattn_layer = attention.RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)
        
        # feed-forward module definition
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        
        
    def forward(self, xs, masks):
        """Encode input sequence.
        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, 1, time).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, 1, time).
        """
        
        xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)
        if isinstance(xs, tuple):
            xs = xs[0]
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks