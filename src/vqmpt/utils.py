""" Useful functions for planning using VQ-MPT models.
"""

import json
from os import path as osp
import torch

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
