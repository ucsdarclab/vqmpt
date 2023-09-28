# Define the vector quantizer module.
# Taken from - https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
#           and https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


class VectorQuantizer(nn.Module):
    """A vector quantizer for storing the dictionary of sample points."""

    def __init__(self, n_e, e_dim, latent_dim):
        """
        :param n_e: Number of elements in the embedding.
        :param e_dim: Size of the latent embedding vector.
        :param latent_dim: Dimension of the encoder vector.
        """
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim

        # Define the linear layer.
        self.input_linear_map = nn.Linear(latent_dim, e_dim)
        self.output_linear_map = nn.Linear(e_dim, latent_dim)

        # Initialize the embedding.
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.batch_norm = nn.BatchNorm1d(self.e_dim, affine=False)

    def forward(self, z, mask):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, num_seq, latent_encoding)

        quantization pipeline:
            1. get encoder output (B, S, E)
            2. flatten input to (B*S, E)
        """
        # flatten input vector
        z_flattened = rearrange(z, "B S E -> (B S) E")
        # pass through the input projection.
        z_flattened = self.input_linear_map(z_flattened)

        # Normalize input vectors.
        z_flattened = F.normalize(z_flattened)
        # Normalize embedding vectors.
        self.embedding.weight.data = F.normalize(self.embedding.weight.data)

        # =========== Since vectors are normalized ==============
        # distances from z to embeddings e_j (z - e)^2 = - e * z
        d = -torch.einsum(
            "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
        )
        # ==============================================================

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q_flattened = self.embedding(min_encoding_indices)

        # Preserve gradients through linear transform also
        z_q_flattened = z_flattened + z_q_flattened - z_flattened.detach()

        # Translate to output encoder shape
        z_q_flattened = self.output_linear_map(z_q_flattened)
        z_q = z_q_flattened.view(z.shape)

        perplexity = None
        min_encodings = None

        return z_q, (perplexity, min_encodings, min_encoding_indices)
