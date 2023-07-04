"""
From the root of Sample Factory repo this can be run as:
python -m sf_examples.train_custom_env_custom_model --algo=APPO --env=my_custom_env_v1 --experiment=example --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m sf_examples.enjoy_custom_env_custom_model --algo=APPO --env=my_custom_env_v1 --experiment=example

"""
from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import gym
import numpy as np
from torch import nn
import minihack
import torch
import torch.nn.functional as F
import math
from sample_factory.utils.utils import str2bool

import nle
from nle import nethack

NUM_CHARS = 256
PAD_CHAR = 0


from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface, register_env
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import nonlinearity
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config, ObsSpace



def calc_conv_output_size(H, W, P, D, K, S, n_layers=2):
    for l in range(n_layers):
        H = math.floor((H + 2*P - D*(K-1) - 1)/S + 1)
        W = math.floor((W + 2*P - D*(K-1) - 1)/S + 1)
    return H * W


class Crop(nn.Module):
    """Helper class for NetHackNet below."""

    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[
            None, :
        ].expand(self.height_target, -1)
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[
            :, None
        ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)
        
        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)

    


# add "TrainingInfoInterface" and "RewardShapingInterface" just to demonstrate how to use them (and for testing)
class MiniHackEnv(gym.Env, TrainingInfoInterface, RewardShapingInterface):
    def __init__(self, full_env_name, cfg, render_mode: Optional[str] = None):
        TrainingInfoInterface.__init__(self)
        self.name = full_env_name  # optional
        self.cfg = cfg
        self.curr_episode_steps = 0

        self.gym_env = gym.make(full_env_name, observation_keys=('glyphs', 'blstats', 'message'))

        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space


        self.render_mode = render_mode


    def reset(self, **kwargs):
        self.curr_episode_steps = 0
        obs = self.gym_env.reset()
        return obs, {}

    def step(self, action):

        # action should be an int here
        assert isinstance(action, (int, np.int32, np.int64))

        obs, reward, done, info = self.gym_env.step(action)

        truncated = self.curr_episode_steps >= self.cfg.custom_env_episode_len

        self.curr_episode_steps += 1

        return obs, reward, done, truncated, dict()

    def render(self):
        pass


    def get_default_reward_shaping(self) -> Dict[str, Any]:
        pass

    def set_reward_shaping(self, reward_shaping: Dict[str, Any], agent_idx: int | slice) -> None:
        pass
    



def make_custom_env_func(full_env_name, cfg=None, _env_config=None, render_mode: Optional[str] = None):
    return MiniHackEnv(full_env_name, cfg, render_mode=render_mode)


def add_extra_params(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument("--encoder_embedding_dim", default=64, type=int, help="embedding dim")
    p.add_argument("--encoder_hidden_dim", default=256, type=int, help="hidden dim for encoder")
    p.add_argument("--encoder_final_activ", default='ln', type=str, help="final activation function for encoder")
    p.add_argument("--encoder_crop_dim", default=16, type=int, help="crop size")
    p.add_argument("--encoder_num_layers", default=2, type=int, help="number of layers")
    p.add_argument("--encoder_msg_model", default='lt_cnn_small', type=str, help="message model")
    p.add_argument("--normalize_intrinsic_rewards", default=True, type=str2bool, help="normalize intrinsic rewards by running std")
    p.add_argument("--intrinsic_reward_coeff", default=1.0, type=float, help="intrinsic_reward coefficient")
    p.add_argument("--custom_env_episode_len", default=400, type=int, help="episode length")


def override_default_params(parser):
    """
    Override default argument values for this family of environments.
    All experiments for environments from my_custom_env_ family will have these parameters unless
    different values are passed from command line.

    """
    parser.set_defaults(
        obs_scale=1.0,
        normalize_input=False,
        normalize_returns=False,
        decorrelate_envs_on_one_worker=False,
        num_envs_per_worker=20,
        batch_size=4096,
        rollout=128,
        recurrence=128,
        save_every_sec=5,
        with_wandb=True,
        intrinsic_reward_global='none',
        wandb_user='mikaelhenaff',
        experiment_summaries_interval=10,
        rnn_size=128,
        rnn_type="lstm",
        train_for_env_steps=5e7,
        exploration_loss_coeff=0.005,
        num_epochs=1,        
    )


class MiniHackEncoder(Encoder):
    """Encoder for MiniHack/NetHack."""

    def __init__(self, cfg, observation_shape, use_crop=True, use_blstats=True, use_glyphs=False, final_activ='none'):
        super().__init__(cfg)

        self.glyph_shape = observation_shape["glyphs"].shape
        self.blstats_size = observation_shape["blstats"].shape[0]

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.cfg = cfg
        self.k_dim = self.cfg.encoder_embedding_dim
        self.h_dim = self.cfg.encoder_hidden_dim
        self.crop_dim = self.cfg.encoder_crop_dim

        self.use_glyphs = use_glyphs
        self.use_crop = use_crop
        self.use_blstats = use_blstats

        self.final_activ = final_activ
        
        if self.final_activ == 'ln':
            self.final_layer_norm = nn.LayerNorm(self.h_dim)
        
        if self.use_crop:
            self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = self.k_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = self.cfg.encoder_num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        out_dim = 0

        if self.use_blstats:
            self.embed_blstats = nn.Sequential(
                nn.Linear(self.blstats_size, self.k_dim),
                nn.ReLU(),
                nn.Linear(self.k_dim, self.k_dim),
                nn.ReLU(),
            )
            out_dim = self.k_dim


        # CNN over full glyph map
        if self.use_glyphs:
            conv_extract = [
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=(F, F),
                    stride=2,
                    padding=P,
                )
                for i in range(L)
            ]

            self.extract_representation = nn.Sequential(
                *interleave(conv_extract, [nn.Sequential(nn.ELU())] * len(conv_extract))
            )
            out_dim += calc_conv_output_size(self.H, self.W, P, 1, F, 2, n_layers=L) * Y

        # CNN crop model.
        if self.use_crop:
            conv_extract_crop = [
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=(F, F),
                    stride=S,
                    padding=P,
                )
                for i in range(L)
            ]

            self.extract_crop_representation = nn.Sequential(
                *interleave(conv_extract_crop, [nn.Sequential(nn.ELU())] * len(conv_extract_crop))
            )
            out_dim += self.crop_dim ** 2 * Y


        self.msg_model = self.cfg.encoder_msg_model
        if self.msg_model == 'lt_cnn_small':
            self.msg_hdim = 32
            self.msg_edim = 16
            self.char_lt = nn.Embedding(
                NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR
            )            
            self.conv1 = nn.Conv1d(
                self.msg_edim, self.msg_hdim, kernel_size=7
            )
            self.conv2_6_fc = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv2
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(3 * self.msg_hdim, self.msg_hdim),
                nn.ReLU(),
                nn.Linear(self.msg_hdim, self.msg_hdim),
            )  # final output -- [ B x h_dim x 5 ]
            out_dim += self.msg_hdim
            
        elif self.msg_model == 'lt_cnn':
            self.msg_hdim = 64
            self.msg_edim = 32
            self.char_lt = nn.Embedding(
                NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR
            )            
            self.conv1 = nn.Conv1d(
                self.msg_edim, self.msg_hdim, kernel_size=7
            )
            # remaining convolutions, relus, pools, and a small FC network
            self.conv2_6_fc = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv2
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv3
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
               # conv4
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
               # conv5
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv6
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(5 * self.msg_hdim, 2 * self.msg_hdim),
                nn.ReLU(),
                nn.Linear(2 * self.msg_hdim, self.msg_hdim),
            )  # final output -- [ B x h_dim x 5 ]
            out_dim += self.msg_hdim
            

        print(f'out_dim: {out_dim}')


        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim)
        )

        self.conv_head_out_size = self.h_dim


        
    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        try:
            return out.reshape(x.shape + (-1,))
        except:
            import pdb; pdb.set_trace()




    def forward(self, obs_dict, flatten=False):
        
        env_outputs = obs_dict

        glyphs = env_outputs["glyphs"].long()
        messages = env_outputs["message"].long()                
        blstats = env_outputs["blstats"].float()

        if flatten:
            glyphs = torch.flatten(glyphs, 0, 1)
            messages = torch.flatten(messages, 0, 1)
            blstats = torch.flatten(blstats, 0, 1)

        batch_size = glyphs.shape[0]


        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]
        # TODO ???
        # coordinates[:, 0].add_(-1)

        # -- [B x F]
        # FIXME: hack to use compatible blstats to before
        # blstats = blstats[:, [0, 1, 21, 10, 11]]

        reps = []
        if self.use_blstats:
            blstats_emb = self.embed_blstats(blstats)
            reps.append(blstats_emb)

        if self.use_crop:
            crop = self.crop(glyphs, coordinates)
            crop_emb = self._select(self.embed, crop)
            crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
            crop_rep = self.extract_crop_representation(crop_emb)
            crop_rep = crop_rep.view(batch_size, -1)
            reps.append(crop_rep)

        if self.use_glyphs:
            glyphs_emb = self._select(self.embed, glyphs)
            glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
            glyphs_rep = self.extract_representation(glyphs_emb)
            glyphs_rep = glyphs_rep.view(batch_size, -1)
            reps.append(glyphs_rep)


        # MESSAGING MODEL
        if self.msg_model != "none":
            # [T x B x 256] -> [T * B x 256]
            messages = messages.view(batch_size, -1)
            if self.msg_model == "lt_cnn":
                # [ T * B x E x 256 ]
                char_emb = self.char_lt(messages).transpose(1, 2)
                char_rep = self.conv2_6_fc(self.conv1(char_emb))
                # TODO: probably too big!

            elif self.msg_model == "lt_cnn_small":
                # [ T * B x E x 256 ]
                messages = messages[:, :128] # most messages are much shorter than 256
                char_emb = self.char_lt(messages).transpose(1, 2)
                char_rep = self.conv2_6_fc(self.conv1(char_emb))
                
            reps.append(char_rep)
        

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st = self.fc(st)

        if self.final_activ == 'sphere':
            st = F.normalize(st, dim=1, p=2)
        elif self.final_activ == 'tanh':
            st = torch.tanh(st)
        elif self.final_activ == 'ln':
            st = self.final_layer_norm(st)
                
        return st
    

    def get_out_size(self) -> int:
        return self.conv_head_out_size


def make_custom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    return MiniHackEncoder(cfg, obs_space, final_activ=cfg.encoder_final_activ)


def register_custom_components():
    MINIHACK_ENVS = [env for env in list(gym.envs.registry.keys()) if ('MiniHack' in env or 'NetHack' in env)]
    for env in MINIHACK_ENVS:
        register_env(env, make_custom_env_func)
    global_model_factory().register_encoder_factory(MiniHackEncoder)


def create_tag(cfg, args):
    cfg_dict = vars(cfg)
    tag = ''
    for k in sorted(args):
        tag += f'{k}={cfg_dict[k]}-'
    tag = tag[:-1]
    
        
    
    

    
def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv, evaluation=evaluation)
    add_extra_params(parser)
    override_default_params(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser, argv)
    return cfg


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
