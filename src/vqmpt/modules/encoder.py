"""Define the Layers
Derived from -
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Layers.py
"""

import numpy as np

from torch import nn
import torch
import torch.utils.checkpoint

from . import sublayers


class EncoderLayer(nn.Module):
    """Single Encoder layer, that consists of a MHA layers and positiion-wise
    feedforward layer.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        Initialize the module.
        :param d_model: Dimension of input/output of this layer
        :param d_inner: Dimension of the hidden layer of hte position-wise feedforward layer
        :param n_head: Number of self-attention modules
        :param d_k: Dimension of each Key
        :param d_v: Dimension of each Value
        :param dropout: Argument to the dropout layer.
        """
        super().__init__()
        self.slf_attn = sublayers.MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.pos_ffn = sublayers.PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout
        )

    def forward(self, enc_input, slf_attn_mask=None):
        """
        The forward module:
        :param enc_input: The input to the encoder.
        :param slf_attn_mask:
        """

        # With Gradient Checking
        enc_output = torch.utils.checkpoint.checkpoint(
            self.slf_attn, enc_input, enc_input, enc_input, slf_attn_mask
        )

        enc_output = self.pos_ffn(enc_output)
        return enc_output


class EncoderLayerPreNorm(nn.Module):
    """Single Encoder layer, that consists of a MHA layers and positiion-wise
    feedforward layer.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        Initialize the module.
        :param d_model: Dimension of input/output of this layer
        :param d_inner: Dimension of the hidden layer of hte position-wise
                        feedforward layer
        :param n_head: Number of self-attention modules
        :param d_k: Dimension of each Key
        :param d_v: Dimension of each Value
        :param dropout: Argument to the dropout layer.
        """
        super(EncoderLayerPreNorm, self).__init__()
        self.slf_attn = sublayers.MultiHeadAttentionPreNorm(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.pos_ffn = sublayers.PositionwiseFeedForwardPreNorm(
            d_model, d_inner, dropout=dropout
        )

    def forward(self, enc_input, slf_attn_mask=None):
        """
        The forward module:
        :param enc_input: The input to the encoder.
        :param slf_attn_mask: mask for self-attn.
        """
        # Without gradient Checking
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class PositionalEncoding(nn.Module):
    """Positional encoding"""

    def __init__(self, d_hid, n_position):
        """
        Intialize the Encoder.
        :param d_hid: Dimesion of the attention features.
        :param n_position: Number of positions to consider.
        """
        super().__init__()
        self.n_pos_sqrt = n_position

        # Not parameters
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """
        Sinusoid position encoding table.
        :param n_position:
        :param d_hid:
        :returns
        """

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.Tensor(sinusoid_table[None, :])

    def forward(self, x):
        """
        Callback function
        :param x:
        """
        pos_enc = self.pos_table[:, : x.size(1)].clone().detach()  # type: ignore
        return x + pos_enc


class EncoderPreNorm(nn.Module):
    """The encoder of the planner."""

    def __init__(
        self,
        n_layers,
        n_heads,
        d_k,
        d_v,
        d_model,
        d_inner,
        c_space_dim,
        dropout,
        n_position,
    ):
        """
        Intialize the encoder.
        :param n_layers: Number of layers of attention and fully connected
                        layer.
        :param n_heads: Number of self attention modules.
        :param d_k: Dimension of each Key.
        :param d_v: Dimension of each Value.
        :param d_model: Dimension of input/output of encoder layer.
        :param d_inner: Dimension of the hidden layers of position wise FFN
        :param c_space_dim: Dimension of the c-space
        :param dropout: The value to the dropout argument.
        :param n_position: Total number of patches the model can handle.
        """
        super().__init__()

        # Embedding
        self.to_embedding = nn.Sequential(
            nn.Linear(c_space_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # Position Encoding.
        # NOTE: Current setup for adding position encoding after patch
        # Embedding.
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayerPreNorm(
                    d_model, d_inner, n_heads, d_k, d_v, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_sequence, returns_attns=False):
        """
        The input of the Encoder should be of dim (b, c, h, w).
        :param input_sequence: Sequence of the trajectories.
        :param returns_attns: If True, the model returns slf_attns at each
                            layer
        """
        enc_slf_attn_list = []
        # Get latent embedding.
        enc_output = self.to_embedding(input_sequence)
        enc_output = self.layer_norm(enc_output)

        # Add position encoding.
        enc_output = self.position_enc(enc_output)

        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=None)

        # Final layer requires a layer-norm
        enc_output = self.layer_norm(enc_output)

        if returns_attns:
            return enc_output, enc_slf_attn_list
        return (enc_output,)
