from typing import Dict, List, Optional

import torch
from gym import spaces
from torch import Tensor, nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.model_utils import ModelModule, create_mlp, model_device, nonlinearity
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, ObsSpace, ActionSpace
from sample_factory.utils.utils import log




def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

    

class InverseDynamicsNet(nn.Module):
    def __init__(self, num_actions, emb_size=128):
        super(InverseDynamicsNet, self).__init__()
        self.num_actions = num_actions 
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * emb_size, 256)), 
            nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(256, self.num_actions))

        
    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=-1)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)
    



    




def default_make_idm_func(action_space: ActionSpace, embedding_dim: int) -> InverseDynamicsNet:
    """Make (most likely convolutional) encoder for image-based observations."""
    inverse_dynamics_model = InverseDynamicsNet(num_actions = action_space.n, emb_size = embedding_dim)
    return inverse_dynamics_model


