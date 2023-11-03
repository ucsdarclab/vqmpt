# Vector Quantized - Motion Planning Transformers

This github repo contains the models, and helper functions for generating sampling distributions using [VQ-MPT](https://sites.google.com/eng.ucsd.edu/vq-mpt/home).

## Installing the package

To install the package, clone this repo to your local machine.

```
git clone https://github.com/jacobjj/vqmpt.git
```

To install the package, go to cloned repo, and run the following command.

```
pip install -e .
```

## Loading models

You can get the pre-trained models for the panda robot from here - *coming soon*

To load the models, use the following:

```
from vqmpt import utils
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
quantizer_model, decoder_model, context_env_encoder, ar_model = utils.get_inference_models(
    decoder_model_folder,
    ar_model_folder,
    device,
    n_e=2048,
    e_dim=8,
)
```

## Getting distributions
