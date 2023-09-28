""" VAE style decoder.
"""
from torch import nn
import torch
import torch.nn.functional as F

from . import sublayers


class DecoderLayer(nn.Module):
    """Compose with three layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        Initialize the Layer
        :param d_model: Dimension of input/output this layer.
        :param d_inner: Dimension of hidden layer of the position wise FFN
        :param n_head: Number of self-attention modules.
        :param d_k: Dimension of each Key.
        :param d_v: Dimension of each Value.
        :param dropout: Argument to the dropout layer.
        """
        super().__init__()
        self.enc_attn = sublayers.MultiHeadAttentionPreNorm(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.pos_ffn = sublayers.PositionwiseFeedForwardPreNorm(
            d_model, d_inner, dropout=dropout
        )

    def forward(self, dec_input, enc_output, dec_enc_attn_mask=None):
        """
        Callback function
        :param dec_input:
        :param enc_output:
        :param slf_attn_mask:
        :param dec_enc_attn_mask:
        """
        dec_output = self.enc_attn(
            dec_input, enc_output, enc_output, dec_mask=dec_enc_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)
        return dec_output


class DecoderPreNormGeneral(nn.Module):
    """Decoder that takes the latent encoding and generates joint samples.
    with non-zero cross correlation variables.
    """

    def __init__(self, e_dim, h_dim, c_space_dim, dropout=0.5):
        """
        :param e_dim: Dimension of the dictionary vectors.
        :param h_dim: Dimension of the feedforward networks hidden vector.
        :param c_space_dim: Dimension of the c-space.
        :param dropout: Dropout value for the fullyconnected layer.
        """
        super().__init__()
        self.pos_ffn = sublayers.PositionwiseFeedForward(e_dim, h_dim, dropout)

        # Layers for returning mean and variance
        self.mu = nn.Sequential(nn.Linear(e_dim, c_space_dim), nn.Tanh())
        self.diag = nn.Sequential(
            nn.Linear(e_dim, c_space_dim),
        )
        self.l = nn.Sequential(
            nn.Linear(e_dim, int(c_space_dim * (c_space_dim - 1) / 2))
        )
        self.register_buffer(
            "l_index", torch.tril_indices(c_space_dim, c_space_dim, offset=-1)
        )
        self.c_space_dim = c_space_dim

    def forward(self, z_q):
        """Returns the decoded mean and variance.
        :param z_q: Latent encoding vectors.
        :returns tuple: mean and diagonal variance vectors.
        """
        z_q = self.pos_ffn(z_q)
        D = torch.diag_embed(F.softplus(self.diag(z_q)))
        L_linear = self.l(z_q)
        L = torch.diag_embed(
            torch.ones(
                (z_q.shape[0], z_q.shape[1], self.c_space_dim),
                device=z_q.device
            )
        )
        L[:, :, self.l_index[0], self.l_index[1]] = L_linear  # type: ignore
        covar = torch.matmul(torch.matmul(L, D), L.transpose(2, 3))
        return (self.mu(z_q) + 1) / 2, covar

    def get_sigma_sqrroot(self, z_q):
        """
        Returns the square root of the covariance matrix.
        :param z_q: a projected dictionary tensor of dimension d_model.
        """
        z_q = self.pos_ffn(z_q)
        D_sqrroot = torch.diag_embed(torch.sqrt(F.softplus(self.diag(z_q))))
        L_linear = self.l(z_q)
        L = torch.diag_embed(
            torch.ones(
                (z_q.shape[0], z_q.shape[1], self.c_space_dim),
                device=z_q.device
            )
        )
        L[:, :, self.l_index[0], self.l_index[1]] = L_linear  # type: ignore
        return L @ D_sqrroot

    def get_mean(self, z_q):
        """
        Returns the mean given projected latent-vector dictonary
        :param z_q: A projected dictionary tensor of dimension d_model
        """
        z_q = self.pos_ffn(z_q)
        return (self.mu(z_q) + 1) / 2
