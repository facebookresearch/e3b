# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn 
from torch.nn import functional as F
import numpy as np
import pdb

from nle import nethack

NUM_CHARS = 256
PAD_CHAR = 0

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module






class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))
    
def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class BasicBlock(nn.Module):
    """
    Residual Network Block
    """
    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


    



class MinigridPolicyNet(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(MinigridPolicyNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0), 
            nn.init.calculate_gain('relu'))
        
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[2], out_channels=32, kernel_size=(3, 3), stride=1, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )
    
        self.fc = nn.Sequential(
            init_(nn.Linear(2048, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        self.core = nn.LSTM(1024, 1024, 2)

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))

        self.policy = init_(nn.Linear(1024, self.num_actions))
        self.baseline = init_(nn.Linear(1024, 1))


    def initial_state(self, batch_size):
        return tuple(torch.zeros(self.core.num_layers, batch_size, 
                                self.core.hidden_size) for _ in range(2))


    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs['partial_obs']
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float()
        
        # -- [unroll_length*batch_size x channels x width x height]
        x = x.permute(0, 3, 1, 2)
        x = self.feat_extract(x)
        x = x.reshape(T * B, -1)
        core_input = self.fc(x)

        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, 
                    action=action), core_state


class MinigridStateEmbeddingNet(nn.Module):
    def __init__(self, observation_shape, use_lstm=False):
        super(MinigridStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape
        self.use_lstm = use_lstm

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[2], out_channels=32, kernel_size=(3, 3), stride=1, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        self.fc = nn.Sequential(
            init_(nn.Linear(2048, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        if self.use_lstm:
            self.core = nn.LSTM(1024, 1024, 2)

    def initial_state(self, batch_size):
        #TODO: we might need to change this
        return tuple(torch.zeros(2, batch_size, 
                                1024) for _ in range(2))
        
    def forward(self, inputs, core_state=(), done=None):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float()
        
        # -- [unroll_length*batch_size x channels x width x height]
        x = x.permute(0, 3, 1, 2)
        x = self.feat_extract(x)
        x = x.reshape(T * B, -1)
        x = self.fc(x)

        if self.use_lstm:
            core_input = x.view(T, B, -1)
            core_output_list = []
            notdone = (~done).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            x = torch.flatten(torch.cat(core_output_list), 0, 1)

        state_embedding = x.view(T, B, -1)
        return state_embedding, core_state


class MinigridMLPEmbeddingNet(nn.Module):
    def __init__(self, hidden_dim=1024):
        super(MinigridMLPEmbeddingNet, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        
    def forward(self, inputs, core_state=()):
        x = inputs
        T, B, *_ = x.shape

        x = self.fc(x)

        state_embedding = x.reshape(T, B, -1)

        return state_embedding, tuple()


class MinigridMLPTargetEmbeddingNet(nn.Module):
    def __init__(self, hidden_dim=1024):
        super(MinigridMLPTargetEmbeddingNet, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        
    def forward(self, inputs, core_state=()):
        x = inputs
        T, B, *_ = x.shape

        try:
            x = self.fc(x)
        except:
            print(self.fc)
            print(x.shape)

        state_embedding = x.reshape(T, B, -1)

        return state_embedding, tuple()


class MinigridInverseDynamicsNet(nn.Module):
    def __init__(self, num_actions, emb_size=128, p_dropout=0.0):
        super(MinigridInverseDynamicsNet, self).__init__()
        self.num_actions = num_actions 
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * emb_size, 256)), 
            nn.ReLU(),
            nn.Dropout(p=p_dropout)
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(256, self.num_actions))

        
    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits
    

class MinigridForwardDynamicsNet(nn.Module):
    def __init__(self, num_actions, hidden_dim=1024):
        super(MinigridForwardDynamicsNet, self).__init__()
        self.num_actions = num_actions 

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
    
        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(hidden_dim + self.num_actions, hidden_dim)), 
            nn.ReLU(), 
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))

        self.fd_out = init_(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, state_embedding, action):
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = torch.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb


def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)


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
    
class NetHackPolicyNet(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_actions,
        use_lstm,
        sphere_norm=0,
        hidden_dim=1024,
        embedding_dim=64,
        crop_dim=9,
        num_layers=5,
        msg_model="lt_cnn"
    ):
        super(NetHackPolicyNet, self).__init__()


        self.register_buffer("reward_sum", torch.zeros(()))
        self.register_buffer("reward_m2", torch.zeros(()))
        self.register_buffer("reward_count", torch.zeros(()).fill_(1e-8))        

        self.glyph_shape = observation_shape["glyphs"].shape
        self.blstats_size = observation_shape["blstats"].shape[0]

        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.sphere_norm = sphere_norm

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = hidden_dim

        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # CNN crop model.
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
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        out_dim = self.k_dim
        # CNN over full glyph map
        out_dim += self.H * self.W * Y

        # CNN crop model.
        out_dim += self.crop_dim ** 2 * Y

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        self.msg_model = msg_model
        if self.msg_model == 'lt_cnn':
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
            
        elif self.msg_model == 'lt_cnn_small':
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
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(5 * self.msg_hdim, 2 * self.msg_hdim),
                nn.ReLU(),
                nn.Linear(2 * self.msg_hdim, self.msg_hdim),
            )  # final output -- [ B x h_dim x 5 ]
            out_dim += self.msg_hdim 

            

        '''
        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )
        '''

        self.fc1 = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )
        

        if self.use_lstm:
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=1)

        self.policy = nn.Linear(self.h_dim, self.num_actions)
        self.baseline = nn.Linear(self.h_dim, 1)



    


    @torch.no_grad()
    def update_running_moments(self, reward_batch):
        """Maintains a running mean of reward."""
        new_count = len(reward_batch)
        new_sum = torch.sum(reward_batch)
        new_mean = new_sum / new_count

        curr_mean = self.reward_sum / self.reward_count
        new_m2 = torch.sum((reward_batch - new_mean) ** 2) + (
            (self.reward_count * new_count)
            / (self.reward_count + new_count)
            * (new_mean - curr_mean) ** 2
        )

        self.reward_count += new_count
        self.reward_sum += new_sum
        self.reward_m2 += new_m2

    @torch.no_grad()
    def get_running_std(self):
        """Returns standard deviation of the running mean of the reward."""
        return torch.sqrt(self.reward_m2 / self.reward_count)        

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs, core_state, decode=False):

#        env_outputs = env_outputs['frame']
        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]

        # -- [T x B x F]
        blstats = env_outputs["blstats"]

        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # -- [B' x F]
        blstats = blstats.view(T * B, -1).float()

        # -- [B x H x W]
        glyphs = glyphs.long()
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]
        # TODO ???
        # coordinates[:, 0].add_(-1)

        # -- [B x F]
        # FIXME: hack to use compatible blstats to before
        # blstats = blstats[:, [0, 1, 21, 10, 11]]

        blstats = blstats.view(T * B, -1).float()
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)

        assert blstats_emb.shape[0] == T * B

        reps = [blstats_emb]

        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)

        # print("crop", crop)
        # print("at_xy", glyphs[:, coordinates[:, 1].long(), coordinates[:, 0].long()])

        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)

        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb)

        # -- [B x K']
        crop_rep = crop_rep.view(T * B, -1)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)
        
        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)


        # MESSAGING MODEL
        if self.msg_model != "none":
            # [T x B x 256] -> [T * B x 256]
            messages = env_outputs["message"].long().view(T * B, -1)
            if self.msg_model == "lt_cnn":
                # [ T * B x E x 256 ]
                char_emb = self.char_lt(messages).transpose(1, 2)
                char_rep = self.conv2_6_fc(self.conv1(char_emb))
            reps.append(char_rep)

        

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st1 = self.fc1(st)
        st = self.fc2(st1)
        if self.sphere_norm == 1:
            st = F.normalize(st, p=2, dim=-1)


        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~env_outputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                try:
                    output, core_state = self.core(input.unsqueeze(0), core_state)
                except:
                    print('self.core')
                    print(core_input)
                    print(self.core)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = st

        # -- [B x A]
        policy_logits = self.policy(core_output)
        # -- [B x A]
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        outputs = dict(policy_logits=policy_logits, baseline=baseline, action=action)
        outputs['policy_hiddens'] = st.detach()
        return (
            outputs,
            core_state,
        )



class NetHackStateEmbeddingNet(nn.Module):
    def __init__(
        self,
        observation_shape,
        use_lstm,
        hidden_dim=1024,
        embedding_dim=64,
        crop_dim=9,
        num_layers=5,
        msg_model="lt_cnn",
        p_dropout=0.0,
    ):
        super(NetHackStateEmbeddingNet, self).__init__()


        self.register_buffer("reward_sum", torch.zeros(()))
        self.register_buffer("reward_m2", torch.zeros(()))
        self.register_buffer("reward_count", torch.zeros(()).fill_(1e-8))        

        self.glyph_shape = observation_shape["glyphs"].shape
        self.blstats_size = observation_shape["blstats"].shape[0]
        self.p_dropout = p_dropout

        self.use_lstm = use_lstm

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = hidden_dim

        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.Sequential(nn.ELU(), nn.Dropout(self.p_dropout))] * len(conv_extract))
        )

        # CNN crop model.
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
            *interleave(conv_extract_crop, [nn.Sequential(nn.ELU(), nn.Dropout(self.p_dropout))] * len(conv_extract))
        )

        self.feat_extract = self.extract_representation

        out_dim = self.k_dim
        # CNN over full glyph map
        out_dim += self.H * self.W * Y

        # CNN crop model.
        out_dim += self.crop_dim ** 2 * Y

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
        )

        self.msg_model = msg_model        
        if msg_model == 'lt_cnn':
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
                nn.Dropout2d(p=self.p_dropout),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv3
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.Dropout2d(p=self.p_dropout),
                # conv4
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.Dropout2d(p=self.p_dropout),
                # conv5
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.Dropout2d(p=self.p_dropout),
                # conv6
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.Dropout2d(p=self.p_dropout),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(5 * self.msg_hdim, 2 * self.msg_hdim),
                nn.ReLU(),
                nn.Dropout(p=self.p_dropout),
                nn.Linear(2 * self.msg_hdim, self.msg_hdim),
            )  # final output -- [ B x h_dim x 5 ]
            out_dim += self.msg_hdim
            

        

        self.fc1 = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        if self.use_lstm:
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=1)



    @torch.no_grad()
    def update_running_moments(self, reward_batch):
        """Maintains a running mean of reward."""
        new_count = len(reward_batch)
        new_sum = torch.sum(reward_batch)
        new_mean = new_sum / new_count

        curr_mean = self.reward_sum / self.reward_count
        new_m2 = torch.sum((reward_batch - new_mean) ** 2) + (
            (self.reward_count * new_count)
            / (self.reward_count + new_count)
            * (new_mean - curr_mean) ** 2
        )

        self.reward_count += new_count
        self.reward_sum += new_sum
        self.reward_m2 += new_m2

    @torch.no_grad()
    def get_running_std(self):
        """Returns standard deviation of the running mean of the reward."""
        return torch.sqrt(self.reward_m2 / self.reward_count)        

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs, core_state):
        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]

        # -- [T x B x F]
        blstats = env_outputs["blstats"]
            
        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # -- [B' x F]
        blstats = blstats.view(T * B, -1).float()

        # -- [B x H x W]
        glyphs = glyphs.long()
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]
        # TODO ???
        # coordinates[:, 0].add_(-1)

        # -- [B x F]
        # FIXME: hack to use compatible blstats to before
        # blstats = blstats[:, [0, 1, 21, 10, 11]]

        blstats = blstats.view(T * B, -1).float()
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)

        assert blstats_emb.shape[0] == T * B

        reps = [blstats_emb]

        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)

        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)

        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb)

        # -- [B x K']
        crop_rep = crop_rep.view(T * B, -1)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)

        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)

        # MESSAGING MODEL
        if self.msg_model != "none":
            # [T x B x 256] -> [T * B x 256]
            messages = env_outputs["message"].long()                
            messages = messages.view(T * B, -1)
            if self.msg_model == "lt_cnn":
                # [ T * B x E x 256 ]
                char_emb = self.char_lt(messages).transpose(1, 2)
                char_rep = self.conv2_6_fc(self.conv1(char_emb))
            reps.append(char_rep)
        

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st1 = self.fc1(st)
        st = self.fc2(st1)
        

        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~env_outputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                try:
                    output, core_state = self.core(input.unsqueeze(0), core_state)
                except:
                    print('self.core')
                    print(core_input)
                    print(self.core)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = st

        core_output = core_output.view(T, B, -1)

        return (
            core_output,
            core_state,
        )




class MarioDoomPolicyNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=True, hidden_size=288):
        super(MarioDoomPolicyNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[0], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )
        self.use_lstm = use_lstm

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                constant_(x, 0))

        if self.use_lstm:
            self.core = nn.LSTM(self.hidden_size, 256, 2)
        else:
            self.core = nn.Linear(self.hidden_size, 256)

        self.policy = init_(nn.Linear(256, self.num_actions))
        self.baseline = init_(nn.Linear(256, 1))


    def initial_state(self, batch_size):
        if self.use_lstm:
            return tuple(torch.zeros(self.core.num_layers, batch_size, 
                                     self.core.hidden_size) for _ in range(2))
        else:
            return ()

    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs['frame']
        T, B, C, W, H = x.shape
        x = x.reshape(T, B, W, H, C)

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        
        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)


        core_input = x.view(T * B, -1)


        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
#            notdone = (~inputs['done'].type(torch.ByteTensor)).float()
            notdone = (~inputs['done']).float()
            if core_input.is_cuda:
                notdone = notdone.cuda()
            t = 0
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                t += 1
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = self.core(core_input)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, 
                    action=action), core_state


class MarioDoomStateEmbeddingNet(nn.Module):
    def __init__(self, observation_shape):
        super(MarioDoomStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[0], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

    def initial_state(self, batch_size):
        return ()
        
    
    def forward(self, inputs, x=None, decode=False):
        # -- [unroll_length x batch_size x height x width x channels]
#        x = inputs
        x = inputs['frame']
        T, B, C, W, H = x.shape
        x = x.reshape(T, B, W, H, C)

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        state_embedding = x.view(T, B, -1)
        
        return state_embedding, None


class MarioDoomForwardDynamicsNet(nn.Module):
    def __init__(self, num_actions, hidden_size=288):
        super(MarioDoomForwardDynamicsNet, self).__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
            
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
    
        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(self.hidden_size + self.num_actions, 256)), 
            nn.ReLU(), 
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))

        self.fd_out = init_(nn.Linear(256, self.hidden_size))

    def forward(self, state_embedding, action):
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = torch.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb


class MarioDoomInverseDynamicsNet(nn.Module):
    def __init__(self, num_actions, hidden_size=288):
        super(MarioDoomInverseDynamicsNet, self).__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * self.hidden_size, 256)), 
            nn.ReLU(), 
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))

        self.id_out = init_(nn.Linear(256, self.num_actions))

        
    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits
    


