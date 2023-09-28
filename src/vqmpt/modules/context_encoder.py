""" Define the context encoder for the network.
"""
from torch import nn

from . import decoder


class ContextEncoder(nn.Module):
    """Converting s/g points to planning context."""

    def __init__(
        self, d_context, d_k, d_v, d_model, d_inner, n_layers, n_heads, dropout=0.1
    ):
        """
        :param d_context: input size of the context map.
        :param d_k: dimension of the key.
        :param d_v: dimension of the value.
        :param d_model: dimension of the latent vector.
        :param d_inner: dimension of fully connected layer.
        :param n_layers: number of self-attention layers.
        :param n_heads: number of heads for self-attention layers.
        :param dropout: dropout for fully connected layer.
        """
        super().__init__()

        # Convert the context to latent embedding
        self.to_latent_embedding = nn.Sequential(
            nn.Linear(d_context, d_inner),
            nn.ReLU(),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self.layer_stack = nn.ModuleList(
            [
                decoder.DecoderLayer(
                    d_model, d_inner, n_heads, d_k, d_v, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, context, env_encoding, env_encoding_mask=None):
        """
        :param context: the s/g pairs or textk embeddings.
        :param env_encoding: Environment encoding from PC++ or FCN layers
        """
        # pass the context through the feed-forward network.
        context_embedding = self.to_latent_embedding(context)

        # Pass the environment embedding through the decoder layer.
        for cross_layer in self.layer_stack:
            context_embedding = cross_layer(
                context_embedding, env_encoding, dec_enc_attn_mask=env_encoding_mask
            )

        context_embedding = self.layer_norm(context_embedding)
        return (context_embedding,)
