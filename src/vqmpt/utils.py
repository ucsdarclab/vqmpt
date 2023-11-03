""" Useful functions for planning using VQ-MPT models.
"""

import json
from os import path as osp
from torch.nn import functional as F

import torch
import numpy as np
import torch_geometric.data as tg_data

from .modules import quantizer
from .modules import decoder
from .modules import autoregressive


def get_inference_models(
    decoder_model_folder,
    ar_model_folder,
    device,
    n_e=2048,
    e_dim=8,
):
    """
    Return the quantizer, decoder, cross-attention, and auto-regressive models.
    :param decoder_model_folder: The folder where the decoder model is stored.
    :param ar_model_folder: The folder where AR model is stored.
    :param device: which device to load the models on.
    :param n_e: Number of dictonary variables to be used.
    :param e_dim: Dimension of the dictionary latent vector.
    :returns tuple: quantizer model, decoder model, environment encoder, ar model
    """
    # Define the decoder model
    with open(osp.join(decoder_model_folder, "model_params.json"), "r") as f:
        dict_model_params = json.load(f)

    decoder_model = decoder.DecoderPreNormGeneral(
        e_dim=dict_model_params["d_model"],
        h_dim=dict_model_params["d_inner"],
        c_space_dim=dict_model_params["c_space_dim"],
    )

    quantizer_model = quantizer.VectorQuantizer(
        n_e=n_e, e_dim=e_dim, latent_dim=dict_model_params["d_model"]
    )
    dec_file = osp.join(decoder_model_folder, "best_model.pkl")
    decoder_checkpoint = torch.load(dec_file, map_location=device)

    # Load model parameters and set it to eval
    for model, state_dict in zip(
        [quantizer_model, decoder_model],
        ["quantizer_state", "decoder_state"],
    ):
        model.load_state_dict(decoder_checkpoint[state_dict])
        model.eval()
        model.to(device)

    # Define the AR + Cross attention model
    with open(osp.join(ar_model_folder, "cross_attn.json"), "r") as f:
        context_env_encoder_params = json.load(f)
    env_params = {
        "d_model": dict_model_params["d_model"],
    }
    context_env_encoder = autoregressive.EnvContextCrossAttModel(
        env_params, context_env_encoder_params, robot="6D"
    )
    # Create the AR model
    with open(osp.join(ar_model_folder, "ar_params.json"), "r") as f:
        ar_params = json.load(f)
    ar_model = autoregressive.AutoRegressiveModel(**ar_params)

    # Load the parameters and set the model to eval
    ar_checkpoint = torch.load(
        osp.join(ar_model_folder, "best_model.pkl"), map_location=device
    )
    for model, state_dict in zip(
        [context_env_encoder, ar_model], ["context_state", "ar_model_state"]
    ):
        model.load_state_dict(ar_checkpoint[state_dict])
        model.eval()
        model.to(device)
    return quantizer_model, decoder_model, context_env_encoder, ar_model


def get_beam_search_path(
    max_length,
    K,
    context_output,
    ar_model,
    quantizer_model,
    goal_index,
    device,
):
    """A beam search function, that stops when any of the paths reaches
    termination.
    :param max_length: Max length to search.
    :param K: Number of paths to keep.
    :param context_output: the tensor ecoding environment information.
    :param ar_model: nn.Model type for the Auto-Regressor.
    :param quantizer_model: For extracting the feature vector.
    :param goal_index: Index used to mark end of sequence
    :param device: device on which to do the processing.
    """

    # Create place holder for input sequences.
    input_seq = torch.ones(K, max_length, 512, dtype=torch.float, device=device) * -1
    quant_keys = torch.ones(K, max_length) * -1
    mask = torch.zeros(K, max_length + 2, device=device)

    ar_model_input_i = torch.cat([context_output.repeat((K, 1, 1)), input_seq], dim=1)
    # mask the start/goal encoding and the prev. sequences.
    mask[:, :3] = 1

    # Get first set of quant_keys
    ar_output = ar_model(ar_model_input_i, mask)
    intial_cost = F.log_softmax(ar_output[:, 2, :], dim=-1)
    # Do not terminate on the final dictionary
    intial_cost[:, goal_index] = -1e9
    path_cost, start_index = intial_cost.topk(k=K, dim=-1)
    start_index = start_index[0]
    path_cost = path_cost[0]
    input_seq[:, 1, :] = quantizer_model.output_linear_map(
        quantizer_model.embedding(start_index)
    )
    quant_keys[:, 0] = start_index
    for i in range(1, max_length - 1):
        ar_model_input_i = torch.cat(
            [context_output.repeat((K, 1, 1)), input_seq], dim=1
        )
        # mask the start/goal encoding and the prev. sequences.
        mask[:, : 3 + i] = 1

        ar_output = ar_model(ar_model_input_i, mask)

        # Get the sequence cost for the next step
        seq_cost = F.softmax(ar_output[:, 2 + i, :], dim=-1)
        # Make self-loops impossible by setting the cost really low
        seq_cost[:, quant_keys[:, i - 1].to(dtype=torch.int64)] = -1e9

        # Get the top set of possible sequences by flattening across batch
        # sizes.
        cur_cost = path_cost[:, None] + seq_cost
        nxt_cost, flatten_index = cur_cost.flatten().topk(K)
        # Reshape back into tensor size to get the approriate batch index and
        # word index.
        new_sequence = torch.as_tensor(
            np.array(np.unravel_index(flatten_index.cpu().numpy(), seq_cost.shape)).T
        )

        # Update previous keys given the current prediction.
        quant_keys[:, :i] = quant_keys[new_sequence[:, 0], :i]
        # Update the current set of keys.
        quant_keys[:, i] = new_sequence[:, 1].to(dtype=torch.float)
        # Update the cost
        path_cost = nxt_cost

        # Break at the first sign of termination
        if (new_sequence[:, 1] == goal_index).any():
            break

        # Select index
        select_index = new_sequence[:, 1] != goal_index

        # Update the input embedding.
        input_seq[select_index, : i + 1, :] = input_seq[
            new_sequence[select_index, 0], : i + 1, :
        ]
        input_seq[select_index, i + 1, :] = quantizer_model.output_linear_map(
            quantizer_model.embedding(new_sequence[select_index, 1].to(device))
        )
    return quant_keys, path_cost, input_seq


def get_search_dist(
    norm_start_n_goal,
    map_data,
    context_encoder,
    decoder_model,
    ar_model,
    quantizer_model,
    num_keys,
    device,
):
    """
    Get the search distribution for a given start and goal state.
    :param norm_start_n_goal: numpy tensor with the normalized start and
        goal
    :param map_data: 3D Point cloud data passed as an numpy array
    :param context_encoder: context encoder model
    :param decoder_model: decoder model to retrive distributions
    :param ar_model: auto-regressive model
    :param quantizer_model: quantizer model
    :param num_keys: Total number of keys in the dictionary
    :param device: device on which to perform torch operations
    :returns (torch.tensor, torch.tensor, float): Returns an array of
        mean and covariance matrix
    """
    # Get the context.
    start_n_goal = torch.as_tensor(
        norm_start_n_goal,
        dtype=torch.float,
    )
    env_input = tg_data.Batch.from_data_list([map_data])
    context_output = context_encoder(env_input, start_n_goal[None, :].to(device))
    # Find the sequence of dict values using beam search
    goal_index = num_keys + 1
    quant_keys, _, input_seq = get_beam_search_path(
        51,
        3,
        context_output,
        ar_model,
        quantizer_model,
        goal_index,
        device,
    )

    reached_goal = torch.stack(torch.where(quant_keys == goal_index), dim=1)
    # Get the distribution.
    if len(reached_goal) > 0:
        # Ignore the zero index, since it is encoding representation of start
        # vector.
        output_dist_mu, output_dist_sigma = decoder_model(
            input_seq[reached_goal[0, 0], 1:reached_goal[0, 1] + 1][None, :]
        )
        dist_mu = output_dist_mu.detach().cpu()
        dist_sigma = output_dist_sigma.detach().cpu()
        # If only a single point is predicted, then reshape the vector to a 2D
        # tensor.
        if len(dist_mu.shape) == 1:
            dist_mu = dist_mu[None, :]
            dist_sigma = dist_sigma[None, :]
        # ========================== append search with goal  ================
        search_dist_mu = torch.zeros((reached_goal[0, 1] + 1, 7))
        search_dist_mu[: reached_goal[0, 1], :6] = dist_mu
        search_dist_mu[reached_goal[0, 1], :] = torch.tensor(norm_start_n_goal[-1])
        search_dist_sigma = torch.diag_embed(torch.ones((reached_goal[0, 1] + 1, 7)))
        search_dist_sigma[: reached_goal[0, 1], :6, :6] = dist_sigma
        search_dist_sigma[reached_goal[0, 1], :, :] = (
            search_dist_sigma[reached_goal[0, 1], :, :] * 0.01
        )
        # ====================================================================
    else:
        search_dist_mu = None
        search_dist_sigma = None
    return search_dist_mu, search_dist_sigma
