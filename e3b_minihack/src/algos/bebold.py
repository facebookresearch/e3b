# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import threading
import time
import timeit
import pprint
import json
import pdb
import copy

import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from src.core import file_writer
from src.core import prof
from src.core import vtrace

import src.models as models
import src.losses as losses

#from src.env_utils import FrameStack
from src.env_utils import FrameStack, Environment, Minigrid2Image
from src.utils import get_batch, log, create_env, create_buffers, act, create_heatmap_buffers

MinigridMLPEmbeddingNet = models.MinigridMLPEmbeddingNet
MinigridMLPTargetEmbeddingNet = models.MinigridMLPTargetEmbeddingNet


NetHackStateEmbeddingNet = models.NetHackStateEmbeddingNet





def learn(actor_model,
          model,
          random_target_network,
          predictor_network,
          actor_encoder,
          encoder,
          batch,
          initial_agent_state, 
          optimizer,
          predictor_optimizer,
          scheduler,
          flags,
          frames=None,
          lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    with lock:
        timings = prof.Timings()
        
        count_rewards = torch.ones((flags.unroll_length, flags.batch_size), 
            dtype=torch.float32).to(device=flags.device)
        # Use the scale of square root N
        count_rewards = batch['episode_state_count'][1:].float().to(device=flags.device)

        timings.reset()
        encoded_states, unused_state = encoder(batch, tuple())
        random_embedding_next, unused_state = random_target_network(encoded_states[1:].detach(), initial_agent_state)
        predicted_embedding_next, unused_state = predictor_network(encoded_states[1:].detach(), initial_agent_state)
        random_embedding, unused_state = random_target_network(encoded_states[:-1].detach(), initial_agent_state)
        predicted_embedding, unused_state = predictor_network(encoded_states[:-1].detach(), initial_agent_state)

        intrinsic_rewards_next = torch.norm(predicted_embedding_next.detach() - random_embedding_next.detach(), dim=2, p=2)
        intrinsic_rewards = torch.norm(predicted_embedding.detach() - random_embedding.detach(), dim=2, p=2)
        intrinsic_rewards = torch.clamp(intrinsic_rewards_next - flags.scale_fac * intrinsic_rewards, min=0)            

        if flags.episodic_bonus_type in ['counts-obs', 'counts-glyphs', 'counts-pos', 'counts-msg', 'counts-img']:
            if flags.count_reward_type == 'ind':
                mask = (count_rewards.to(flags.device) == 1).float()
                intrinsic_rewards *= mask
            elif flags.count_reward_type == 'isqrt':
                intrinsic_rewards *= count_rewards
            elif flags.count_reward_type == 'const':
                pass
#            intrinsic_rewards *= flags.intrinsic_reward_coef            
        else:
            raise NotImplementedError("invalid episodic_bonus_type")
            

        timings.time('int. reward comp')

        timings.reset()
        
        num_samples = flags.unroll_length * flags.batch_size
        actions_flat = batch['action'][1:].reshape(num_samples).cpu().detach().numpy()
        rnd_loss = flags.rnd_loss_coef * \
                losses.compute_rnd_loss(predicted_embedding_next, random_embedding_next.detach())

#        print(actions_flat)
            
        learner_outputs, unused_state = model(batch, initial_agent_state)#, decode=False)        

        bootstrap_value = learner_outputs['baseline'][-1]

        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }
        
        rewards = batch['reward']


        if flags.reward_norm == 'int':
            model.update_running_moments(intrinsic_rewards)
            std = model.get_running_std()
            if std > 0:
                intrinsic_rewards /= std
        elif flags.reward_norm == 'ext':
            model.update_running_moments(rewards)
            std = model.get_running_std()
            if std > 0:
                rewards /= std


        
        if flags.no_reward:
            total_rewards = intrinsic_rewards
        else:            
            total_rewards = rewards + intrinsic_rewards * flags.intrinsic_reward_coef
            
        if flags.clip_rewards == 1:
            clipped_rewards = torch.clamp(total_rewards, -1, 1)
        else:
            clipped_rewards = total_rewards

            
            
        
        discounts = (~batch['done']).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value)

        pg_loss = losses.compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                               batch['action'],
                                               vtrace_returns.pg_advantages)
        baseline_loss = flags.baseline_cost * losses.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = flags.entropy_cost * losses.compute_entropy_loss(
            learner_outputs['policy_logits'])

        total_loss = pg_loss + baseline_loss + entropy_loss + rnd_loss #+ decoder_loss


        episode_returns = batch['episode_return'][batch['done']]
        stats = {
            'mean_episode_return': torch.mean(episode_returns).item(),
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'rnd_loss': rnd_loss.item(),
            'mean_rewards': torch.mean(rewards).item(),
            'mean_intrinsic_rewards': torch.mean(intrinsic_rewards).item(),
            'mean_total_rewards': torch.mean(total_rewards).item(),
        }
        
        optimizer.zero_grad()
        predictor_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        nn.utils.clip_grad_norm_(predictor_network.parameters(), flags.max_grad_norm)
        optimizer.step()
        predictor_optimizer.step()
#        scheduler.step()

        timings.time('policy update')
        timings.reset()
        
        timings.time('encoder update')

#        print(timings.summary())

        actor_model.load_state_dict(model.state_dict())
        actor_encoder.load_state_dict(encoder.state_dict())
        return stats, None#decoder_logits.detach().cpu()



    
    

def train(flags):

    xpid = ''
    xpid += f'env_{flags.env}'
    xpid += f'-model_{flags.model}'
    xpid += f'-bt_{flags.episodic_bonus_type}'
    xpid += f'-cr_{flags.count_reward_type}'
    xpid += f'-lr_{flags.learning_rate}'
    xpid += f'-cl_{flags.clip_rewards}'
    xpid += f'-rn_{flags.reward_norm}'
    xpid += f'-entropy_{flags.entropy_cost}'
    xpid += f'-intweight_{flags.intrinsic_reward_coef}'
    xpid += f'-scalefac_{flags.scale_fac}'
    xpid += f'-seed_{flags.seed}'

    flags.xpid = xpid
        
    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )

    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                         'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    if not flags.disable_cuda and torch.cuda.is_available():
        log.info('Using CUDA.')
        flags.device = torch.device(f'cuda:{flags.device}')
    else:
        log.info('Not using CUDA.')
        flags.device = torch.device('cpu')

    env = create_env(flags)

    if flags.num_input_frames > 1:
        env = FrameStack(env, flags.num_input_frames)  
        
    if 'MiniHack' in flags.env:
        model = models.NetHackPolicyNet(env.observation_space, env.action_space.n, flags.use_lstm, hidden_dim=flags.hidden_dim, sphere_norm=flags.sphere_norm, msg_model=flags.msg_model)
        random_target_network = MinigridMLPTargetEmbeddingNet(hidden_dim=flags.hidden_dim).to(device=flags.device) 
        predictor_network = MinigridMLPEmbeddingNet(hidden_dim=flags.hidden_dim).to(device=flags.device) 
        encoder = NetHackStateEmbeddingNet(env.observation_space, False, hidden_dim=flags.hidden_dim, msg_model=flags.msg_model) #do not use LSTM
    elif 'Vizdoom' in flags.env:
        model = models.MarioDoomPolicyNet(env.observation_space.shape, env.action_space.n)
        random_target_network = MinigridMLPTargetEmbeddingNet(hidden_dim=flags.hidden_dim).to(device=flags.device) 
        predictor_network = MinigridMLPEmbeddingNet(hidden_dim=flags.hidden_dim).to(device=flags.device) 
        encoder = models.MarioDoomStateEmbeddingNet(env.observation_space.shape).to(flags.device)
    elif 'MiniWorld' in flags.env:
        model = models.MarioDoomPolicyNet(env.observation_space.shape, env.action_space.n, hidden_size=flags.hidden_dim)
        random_target_network = MinigridMLPTargetEmbeddingNet(hidden_dim=flags.hidden_dim).to(device=flags.device) 
        predictor_network = MinigridMLPEmbeddingNet(hidden_dim=flags.hidden_dim).to(device=flags.device) 
        encoder = models.MarioDoomStateEmbeddingNet(env.observation_space.shape).to(flags.device)
    else:
        raise Exception('Only MiniHack is suppported Now!')


    buffers = create_buffers(env.observation_space, model.num_actions, flags)
    model.share_memory()
    encoder.share_memory()
    
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context('fork')
    free_queue = ctx.Queue()
    full_queue = ctx.Queue()

    episode_state_count_dict = dict()
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, model, None, buffers, 
                  episode_state_count_dict, initial_agent_state_buffers, flags))
        actor.start()
        actor_processes.append(actor)

    if 'MiniHack' in flags.env:
        learner_model = models.NetHackPolicyNet(env.observation_space, env.action_space.n, flags.use_lstm, hidden_dim=flags.hidden_dim, msg_model=flags.msg_model).to(flags.device)
        learner_encoder = NetHackStateEmbeddingNet(env.observation_space, False, hidden_dim=flags.hidden_dim, msg_model=flags.msg_model).to(device=flags.device)
    elif 'Vizdoom' in flags.env:
        learner_model = models.MarioDoomPolicyNet(env.observation_space.shape, env.action_space.n).to(flags.device)
        learner_encoder = models.MarioDoomStateEmbeddingNet(env.observation_space.shape).to(flags.device)
    elif 'MiniWorld' in flags.env:
        learner_model = models.MarioDoomPolicyNet(env.observation_space.shape, env.action_space.n, hidden_size=flags.hidden_dim).to(flags.device)
        learner_encoder = models.MarioDoomStateEmbeddingNet(env.observation_space.shape).to(flags.device)
    else:
        raise Exception('Only MiniHack is suppported Now!')
    learner_encoder.load_state_dict(encoder.state_dict())

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    predictor_optimizer = torch.optim.Adam(
        predictor_network.parameters(), 
        lr=flags.predictor_learning_rate)

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger('logfile')
    stat_keys = [
        'total_loss',
        'mean_episode_return',
        'pg_loss',
        'baseline_loss',
        'entropy_loss',
        'rnd_loss',
        'mean_rewards',
        'mean_intrinsic_rewards',
        'mean_total_rewards',
    ]

    logger.info('# Step\t%s', '\t'.join(stat_keys))

    frames, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        while frames < flags.total_frames:
            timings.reset()
            batch, agent_state = get_batch(free_queue, full_queue, buffers, 
                initial_agent_state_buffers, flags, timings)
            stats, decoder_logits = learn(model, learner_model, random_target_network, predictor_network,
                                          encoder, learner_encoder, batch, agent_state, optimizer, 
                                          predictor_optimizer, scheduler, flags, frames=frames)
            timings.time('learn')
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                

        if i == 0:
            log.info('Batch and learn: %s', timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []    
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,))
        thread.start()
        threads.append(thread)


    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        checkpointpath = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid,'model.tar')))
        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'frames': frames,
            'model_state_dict': model.state_dict(),
            'encoder': encoder.state_dict(),
            'random_target_network_state_dict': random_target_network.state_dict(),
            'predictor_network_state_dict': predictor_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'predictor_optimizer_state_dict': predictor_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'flags': vars(flags),
        }, checkpointpath)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
#                save_heatmap(frames)
                last_checkpoint_time = timer()

            fps = (frames - start_frames) / (timer() - start_time)
            
            if stats.get('episode_returns', None):
                mean_return = 'Return per episode: %.1f. ' % stats[
                    'mean_episode_return']
            else:
                mean_return = ''

            total_loss = stats.get('total_loss', float('inf'))
            if stats:
                log.info('After %i frames: loss %f @ %.1f fps. Mean Return %.1f. \n Stats \n %s', \
                        frames, total_loss, fps, stats['mean_episode_return'], pprint.pformat(stats))

    except KeyboardInterrupt:
        return  
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint(frames)
    plogger.close()

