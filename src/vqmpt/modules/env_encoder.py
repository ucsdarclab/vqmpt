""" Defining layers for converting maps to latent encodings.
"""
import torch
from torch import nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange
import torch_geometric.nn as tgnn


class PositionalEncoding(nn.Module):
    """Positional encoding"""

    def __init__(self, d_hid, n_position):
        """
        Intialize the Encoder.
        :param d_hid: Dimesion of the attention features.
        :param n_position: Number of positions to consider.
        :param train_shape: The 2D shape of the training model.
        """
        super(PositionalEncoding, self).__init__()
        self.n_pos_sqrt = int(np.sqrt(n_position))
        # Not a parameter
        self.register_buffer("hashIndex", self._get_hash_table(n_position))
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_hash_table(self, n_position):
        """
        A simple table converting 1D indexes to 2D grid.
        :param n_position: The number of positions on the grid.
        """
        return rearrange(
            torch.arange(n_position),
            "(h w) -> h w",
            h=int(np.sqrt(n_position)),
            w=int(np.sqrt(n_position)),
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """
        Sinusoid position encoding table.
        :param n_position:
        :param d_hid:
        :returns
        """
        # TODO: make it with torch instead of numpy
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
        return torch.FloatTensor(sinusoid_table[None, :])

    def forward(self, x, conv_shape):
        """
        Callback function
        :param x:
        """
        selectIndex = rearrange(
            self.hashIndex[: conv_shape[0], : conv_shape[1]], "h w -> (h w)"  # type: ignore
        )
        return x + torch.index_select(self.pos_table, dim=1, index=selectIndex)  # type: ignore


# Encoder for environment
class EnvEncoder(nn.Module):
    """The environment encoder of the planner."""

    def __init__(self, d_model, dropout, n_position):
        """
        Intialize the encoder.
        :param n_layers: Number of layers of attention and fully connected layer.
        :param n_heads: Number of self attention modules.
        :param d_k: Dimension of each Key.
        :param d_v: Dimension of each Value.
        :param d_model: Dimension of input/output of encoder layer.
        :param d_inner: Dimension of the hidden layers of position wise FFN
        :param dropout: The value to the dropout argument.
        :param n_position: Total number of patches the model can handle.
        :param train_shape: The shape of the output of the patch encodings.
        """
        super().__init__()
        # Convert the image to and input embedding.
        # NOTE: This is one place where we can add convolution networks.
        # Convert the image to linear model

        # NOTE: Padding of 3 is added to the final layer to ensure that
        # the output of the network has receptive field across the entire map.
        # NOTE: pytorch doesn't have a good way to ensure automatic padding.
        # This allows only for a select few map sizes to be solved using this
        # method.
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, d_model, kernel_size=5, stride=5, padding=3),
        )

        self.reorder_dims = Rearrange("b c h w -> b (h w) c")
        # Position Encoding.
        # NOTE: Current setup for adding position encoding after patch
        # Embedding.
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_map, returns_attns=False):
        """
        The input of the Encoder should be of dim (b, c, h, w).
        :param input_map: The input map for planning.
        :param returns_attns: If True, the model returns slf_attns at
            each layer
        """
        enc_output = self.to_patch_embedding(input_map)
        conv_map_shape = enc_output.shape[-2:]
        enc_output = self.reorder_dims(enc_output)

        enc_output = self.position_enc(enc_output, conv_map_shape)

        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)
        return enc_output


# Point cloud encoder
class SAModule(nn.Module):
    """The set abstraction layer"""

    def __init__(self, ratio, r, channels):
        """Initialization of the model.
        :param ratio: Amount of points dropped
        :param r: the radius of grouping
        :param channels: Shared weights for each latent vectors
        """
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(c, channels[i + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(channels[i + 1]),
                )
                for i, c in enumerate(channels[:-1])
            ]
        )
        # NOTE: "Here, we do not really want to add self-loops to the graph as
        # we are operating in bipartite graphs. The real "self-loop" is already
        # added to tgnn.PointConv by the radius call."
        # Ref: https://github.com/pyg-team/pytorch_geometric/issues/2558
        self.conv = tgnn.PointNetConv(local_nn=mlp, add_self_loops=False).jittable()

    def forward(self, x, pos, batch):
        """Forward propogation of the model."""
        # Reduce the density of point cloud by farthest point sampling
        # random_start=False, This is to ensure origin is added to the graph
        idx = tgnn.fps(pos, batch, ratio=self.ratio, random_start=False)
        # row - indexes for y
        # col - indexes for x
        row, col = tgnn.radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        # readjust the indexes for creating edge_index.
        newRow = idx[row]
        edge_index = torch.stack([col, newRow], dim=0)
        x = self.conv(x, pos, edge_index)
        pos, batch = pos[idx], batch[idx]
        return x[idx], pos, batch, idx


class FeatureExtractor(torch.nn.Module):
    """Extract features from using PointNet++ architecture"""

    def __init__(self, d_model):
        """Initialize the network.
        :param input_dim: dimension of the point cloud data point.
        :param d_model: dimension of the final latent layer
        """
        super(FeatureExtractor, self).__init__()
        self.sa1_module = SAModule(0.75, 0.2, channels=[3 + 3, 64, 128])
        self.sa2_module = SAModule(0.75, 0.4, channels=[128 + 3, 256, d_model])

    def forward(self, data):
        """
        :param data: An object of type torch_geometric.data.Batch
        :returns tuple: (latent_vector, tensor_point, batch)
        """
        allIndex = torch.arange(data.pos.shape[0], device=data.pos.device)
        *h_pos_batch, idx = self.sa1_module(data.pos, data.pos, data.batch)
        allIndex = allIndex[idx]
        *h_pos_batch, idx = self.sa2_module(*h_pos_batch)
        allIndex = allIndex[idx]
        return h_pos_batch, allIndex
