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

You can get the pre-trained models for the panda robot from here - [Panda Models](https://drive.google.com/file/d/1B0KVBxYBi0fCQcvagponF6j_2TikZfN7/view?usp=sharing)

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
To get the distribution, you can use the `get_search_dist` in `utils.py`. Normalize the start and goal configurations between [0, 1], and stack them together to form a 2*n_dim.

```
search_dist_mu, search_dist_sigma = utils.get_search_dist(
    norm_start_n_goal,
    depth_points,
    context_encoder,
    decoder_model,
    ar_model,
    quantizer_model,
    num_keys=2048,
    device=device,
)
```

## Sampling from the distribution

If you are using the distribution with OMPL, here is an example of writing a custom sampling function.

```
from ompl import base as ob

class StateSamplerRegion(ob.StateSampler):
    '''A class to sample robot joints from a given joint configuration.
    '''
    def __init__(self, space, qMin=None, qMax=None, dist_mu=None, dist_sigma=None):
        '''
        If dist_mu is None, then set the sampler as a uniform sampler.
        :param space: an object of type ompl.base.Space
        :param qMin: np.array of minimum joint bound
        :param qMax: np.array of maximum joint bound
        :param region: np.array of points to sample from
        '''
        super(StateSamplerRegion, self).__init__(space)
        self.name_ ='region'
        self.q_min = qMin
        self.q_max = qMax
        if dist_mu is None:
            self.X = None
            self.U = stats.uniform(np.zeros_like(qMin), np.ones_like(qMax))
        else:
            self.seq_num = dist_mu.shape[0]
            self.X = MultivariateNormal(dist_mu, dist_sigma)

                       
    def get_random_samples(self):
        '''Generates a random sample from the list of points
        '''
        index = 0
        random_samples = np.random.permutation(self.X.sample()*(self.q_max-self.q_min)+self.q_min)

        while True:
            yield random_samples[index, :]
            index += 1
            if index==self.seq_num:
                random_samples = np.random.permutation(self.X.sample()*(self.q_max-self.q_min)+self.q_min)
                index = 0
                
    def sampleUniform(self, state):
        '''Generate a sample from uniform distribution or key-points
        :param state: ompl.base.Space object
        '''
        if self.X is None:
            sample_pos = ((self.q_max-self.q_min)*self.U.rvs()+self.q_min)[0]
        else:
            sample_pos = next(self.get_random_samples())
        for i, val in enumerate(sample_pos):
            state[i] = float(val)
        return True
```