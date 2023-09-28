""" The autoregressive model for predicting steps.
"""
import torch.nn as nn
import torch_geometric.utils as tg_utils

from . import encoder
from . import env_encoder
from . import context_encoder


class AutoRegressiveModel(nn.Module):
    """Get the encoder input and convert it to set of logits values."""

    def __init__(
        self, d_k, d_v, d_model, d_inner, n_layers, n_heads, num_keys, dropout=0.1
    ):
        """
        :param d_k: dimension of the key.
        :param d_v: dimension of the value.
        :param d_inner: dimension of the latent vector.
        :param d_model: dimension of the latent vector.
        :param n_layers: Number of self-attention layers.
        :param n_heads: number of heads for self-attention layers.
        :param dropout: dropout for fully connected layer.
        """
        super().__init__()

        self.layer_stack = nn.ModuleList(
            [
                encoder.EncoderLayerPreNorm(
                    d_model, d_inner, n_heads, d_k, d_v, dropout
                )
                for _ in range(n_layers)
            ]
        )

        # Add layer norm to the final layer
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Implement logit function.
        self.class_pred = nn.Linear(d_model, num_keys)

    def forward(self, enc_output, slf_attn_mask=None):
        """
        The forward module:
        :param enc_input: the i/p to the encoder.
        :param slf_attn_mask: mask for the self-attn.
        """
        for attn_layer in self.layer_stack:
            enc_output = attn_layer(enc_output, slf_attn_mask)

        # Add layer normalization to the final layer
        enc_output = self.layer_norm(enc_output)

        # pass through logit function.
        enc_output = self.class_pred(enc_output)
        return enc_output


class EnvContextCrossAttModel(nn.Module):
    """Given the context and environment model, return the cross attention model."""

    def __init__(self, env_params, context_params, robot="2D"):
        """
        :param env_params: A dictionary with values for the following keys for the envirnoment encoder
            {n_layers, n_heads, d_k, d_v, d_model, d_inner, dropout, n_position}
        :param context_params: A dict with values for the following keys for the context encoder.
            {}
        """
        super().__init__()

        # Define Environment model.
        if robot == "2D":
            self.env_encoder = env_encoder.EnvEncoder(**env_params)  # type: ignore

        if robot == "6D" or robot == "14D" or robot == "7D":
            self.env_encoder = env_encoder.FeatureExtractor(**env_params)  # type: ignore

        self.robot = robot

        # Translate context embedding and do cross-attention.
        self.context_encoder = context_encoder.ContextEncoder(**context_params)  # type: ignore

    def forward(self, env_input, start_goal_input):
        cross_encoding_output = None
        # Pass the input through the encoder.
        if self.robot == "2D":
            env_encoding_output = self.env_encoder(env_input)
            # Take the cross attention model.
            (cross_encoding_output,) = self.context_encoder(
                start_goal_input, env_encoding_output
            )

        if self.robot == "6D" or self.robot == "14D" or self.robot == "7D":
            (h, _, batch), _ = self.env_encoder(env_input)
            env_encoding_output, dec_mask = tg_utils.to_dense_batch(h, batch)
            # Take the cross attention model
            (cross_encoding_output,) = self.context_encoder(
                start_goal_input, env_encoding_output, env_encoding_mask=dec_mask
            )

        return cross_encoding_output
