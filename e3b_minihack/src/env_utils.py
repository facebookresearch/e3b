# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gym 
import torch 
from collections import deque, defaultdict
from gym import spaces
import numpy as np
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
import pdb


def _format_observation_vizdoom(obs):
    obs = torch.tensor(obs)
    obs = obs.view((1, 1) + obs.shape)
    return obs


def _format_observations_nethack(observation, keys=("glyphs", "blstats", "message")):
    observations = {}
    if 'state_visits' in observation.keys():
        keys += ('state_visits',)

    for key in keys:
        entry = observation[key]
        entry = torch.from_numpy(entry)
        entry = entry.view((1, 1) + entry.shape)  # (...) -> (T,B,...).
        observations[key] = entry
    return observations



class Environment:
    def __init__(self, gym_env, fix_seed=False, env_seed=1):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.episode_win = None
        self.fix_seed = fix_seed
        self.env_seed = env_seed

    def get_partial_obs(self):
        return self.gym_env.env.env.gen_obs()['image']

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        self.episode_win = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        if self.fix_seed:
            self.gym_env.seed(seed=self.env_seed)
        obs = self.gym_env.reset()
        if type(obs) is dict:
            initial_frame = _format_observations_nethack(obs)
            partial_obs = None
            carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])
            initial_frame.update(
                reward=initial_reward,
                done=initial_done,
                episode_return=self.episode_return,
                episode_step=self.episode_step,
            )
            return initial_frame
        else:
            initial_frame = _format_observation_vizdoom(self.gym_env.reset())

            return dict(
                frame=initial_frame,
                reward=initial_reward,
                done=initial_done.bool(),
                episode_return=self.episode_return,
                episode_step=self.episode_step,
            )
        
    def step(self, action):
        if not isinstance(action, int):
            action = action.item()
            
        frame, reward, done, _ = self.gym_env.step(action)

        self.episode_step += 1
        episode_step = self.episode_step

        self.episode_return += reward
        episode_return = self.episode_return 

        if done and reward > 0:
            self.episode_win[0][0] = 1 
        else:
            self.episode_win[0][0] = 0 
        episode_win = self.episode_win 
        
        if done:
            if self.fix_seed:
                self.gym_env.seed(seed=self.env_seed)
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
            self.episode_win = torch.zeros(1, 1, dtype=torch.int32)

        if type(frame) is dict:
            frame = _format_observations_nethack(frame)
            reward = torch.tensor(reward).view(1, 1)
            done = torch.tensor(done).view(1, 1)        
            carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])

            
            frame.update(
                reward=reward,
                done=done,
                episode_return=episode_return,
                episode_step = episode_step,
            )
            return frame
                
        else:
            frame = _format_observation_vizdoom(frame)
            reward = torch.tensor(reward).view(1, 1)
            done = torch.tensor(done).view(1, 1).bool()
            
            return dict(
                frame=frame,
                reward=reward,
                done=done,
                episode_return=episode_return,
                episode_step = episode_step,
                episode_win = episode_win,
            )

            
    def close(self):
        self.gym_env.close()


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


