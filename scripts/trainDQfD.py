import sys
# import pickle
import os
import numpy as np
from collections import namedtuple
from itertools import count
import random
from torch.utils.tensorboard import SummaryWriter
from utils.image2image_wrapper import translate_state

import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim


from utils.ReplayBuffer import PrioritizedReplayBuffer

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def recollect_experience(env,q_func):
    img_h, img_w, img_c = env.observation_space.shape
    input_arg = img_c
    num_actions = env.action_space.n

    def process_observation(obs):
        return np.array(obs)[None][0]

    def load_model(model):
        if os.path.isfile('mario_Q_params_2357.pkl'):
            print('Load Q parametets ...')
            model.load_state_dict(torch.load('mario_Q_params_2357.pkl'))
        return model

    def epsilon_greedy_action(model, obs):
        sample = random.random()
        eps = 0.05
        if sample > eps:

            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)/255.0

            return model(Variable(obs)).data.max(1)[1].view(-1, 1).cpu()

        else:
            return torch.IntTensor([[random.randrange(num_actions)]]).cpu()
    
    Q = q_func(input_arg, num_actions).type(dtype)
    Q = load_model(Q)
    experience = []
    while(len(experience)<=6000):
        act_exp = []
        done = False
        acum_rew = 0
        last_obs = env.reset()
        last_obs = process_observation(last_obs)
        while not done:
            action = epsilon_greedy_action(Q, translate_state(last_obs).transpose(2, 0, 1)).numpy()[0, 0]
            obs, reward, done, _ = env.step(action)
            obs = process_observation(obs)
            acum_rew += reward
            act_exp.append((last_obs,action,reward,obs,done))
            last_obs = obs    
            if done:
                if acum_rew > 500.0:
                    print(acum_rew)
                    experience.extend(act_exp)
                    print(len(experience))
                else:
                    print("FAILED EPISODE")
    print("DONE Recollect data")
    return experience
    

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

def dqfd_learn(
    env,
    q_func,
    optimizer_spec,
    exploration,
    annelation,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=500,
    learning_freq=4,
    alpha=0.6,
    lr=0.001,
        target_update_freq=10000):

    img_h, img_w, img_c = env.observation_space.shape
    input_arg = img_c
    num_actions = env.action_space.n

    def load_model(model):
        if os.path.isfile('mario_Q_params_2357.pkl'):
            print('Load Q parametets ...')
            model.load_state_dict(torch.load('mario_Q_params_2357.pkl'))
        return model

    def epsilon_greedy_action(model, obs):
        sample = random.random()
        eps = 0.05
        if sample > eps:

            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)/255.0

            return model(Variable(obs)).data.max(1)[1].view(-1, 1).cpu()

        else:
            return torch.IntTensor([[random.randrange(num_actions)]]).cpu()

    def create_networks():
        return q_func(input_arg, num_actions).type(dtype), q_func(input_arg, num_actions).type(dtype)

    def load_previous_models(model, target):
        if os.path.isfile('nets/mario_Q_params_1282.pkl'):
            print('Load Q parametets ...')
            model.load_state_dict(torch.load('nets/mario_Q_params_1282.pkl'))
        if os.path.isfile('nets/mario_target_Q_params_1282.pkl'):
            print('Load target Q parameters ...')
            target.load_state_dict(torch.load('nets/mario_target_Q_params_1282.pkl'))
        return model, target

    def process_observation(obs):
        return np.array(obs)[None][0]

    def get_batch2(t):
        # Use the replay buffer to sample a batch of transitions
        # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, weights, indxes = replay_buffer.sample(batch_size, annelation.value(t))
        # Convert numpy nd_array to torch variables for calculation
        obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
        act_batch = Variable(torch.from_numpy(act_batch).long())
        rew_batch = Variable(torch.from_numpy(rew_batch).type(dtype))
        next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)
        weights = Variable(torch.from_numpy(weights).type(dtype))

        if USE_CUDA:
            act_batch = act_batch.cuda()
            rew_batch = rew_batch.cuda()

        return obs_batch, act_batch, rew_batch, next_obs_batch, not_done_mask, weights, indxes

    def get_batch():
        # Use the replay buffer to sample a batch of transitions
        # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, weights, indxes = replay_buffer.sample(32, 0.4)
        # Convert numpy nd_array to torch variables for calculation
        obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
        act_batch = Variable(torch.from_numpy(act_batch).long().unsqueeze(1))
        rew_batch = Variable(torch.from_numpy(rew_batch).type(dtype))
        next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)
        weights = Variable(torch.from_numpy(weights).type(dtype))

        if USE_CUDA:
            obs_batch = obs_batch.cuda()
            act_batch = act_batch.cuda()
            rew_batch = rew_batch.cuda()
            next_obs_batch = next_obs_batch.cuda()
            not_done_mask = not_done_mask.cuda()

        return obs_batch, act_batch, rew_batch, next_obs_batch, not_done_mask, weights, indxes

    def save_info(LOG_EVERY_N_STEPS, t, mean_episode_reward, best_mean_episode_reward):
        episode_rewards = env.get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])

        if(mean_episode_reward > best_mean_episode_reward and mean_episode_reward > 0 and len(episode_rewards) > 100):
            if int(mean_episode_reward) > 700:
                print("best reward saved: %f", mean_episode_reward)
                torch.save(Q.state_dict(), 'nets/mario_Q_params_{}.pkl'.format((int(mean_episode_reward))))
                torch.save(target_Q.state_dict(), 'nets/mario_target_Q_params_{}.pkl'.format((int(mean_episode_reward))))
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)            

        # Statistic["mean_episode_rewards"].append(mean_episode_reward)
        # Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            # Dump statistics to pickle
            # with open('statistics.pkl', 'wb') as f:
            #   pickle.dump(Statistic, f)
            #    print("Saved to %s" % 'statistics.pkl')
        return mean_episode_reward, best_mean_episode_reward

    def train_step(model, target, num_param_updates, t, writer):
        obs_batch, act_batch, rew_batch, next_obs_batch, not_done_mask, weights, indxes = get_batch2(t)

        # Compute current Q value, q_func takes only state and output value for every state-action pair
        # We choose Q based on action taken.
        current_Q_values = model(obs_batch).gather(1, act_batch.view(-1, 1))

        """
        # DQN
        # Compute next Q value based on which action gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = target_Q(next_obs_batch).detach().max(1)[0].view(-1, 1)
        next_Q_values = not_done_mask.view(-1, 1) * next_max_q
        """
        next_argmax_action = model(next_obs_batch).max(1)[1].view(-1, 1)
        next_q = target(next_obs_batch).detach().gather(1, next_argmax_action)
        next_Q_values = not_done_mask.view(-1, 1) * next_q
        # Compute the target of the current Q values
        target_Q_values = rew_batch.view(-1, 1) + (gamma * next_Q_values)

        # compute td error
        td_error = current_Q_values.data.cpu().numpy()-target_Q_values.data.cpu().numpy()

        """
        # Compute Bellman error
        bellman_error = target_Q_values - current_Q_values
        # clip the bellman error between [-1 , 1]
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        # Note: clipped_bellman_delta * -1 will be right gradient
        d_error = clipped_bellman_error * -1.0
        # Clear previous gradients before backward pass
        optimizer.zero_grad()
        # run backward pass
        current_Q_values.backward(d_error.data)
        """
        loss = F.mse_loss(current_Q_values, target_Q_values)
        loss = loss*weights
        loss = torch.mean(loss)
        writer.add_scalar('Loss', loss, t)
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp(-1, 1)
        # Perfom the update
        optimizer.step()
        replay_buffer.update_priorities(indxes, abs(td_error)+1e-09)
        return num_param_updates+1

   
    
    def optimize_dqfd(model, step, writer):
        obs_batch, act_batch, rew_batch, next_obs_batch, not_done_mask, _, _ = get_batch()
        
        q_vals = model(obs_batch)
        #print(q_vals.shape)
        #print(act_batch.shape)
        state_action_values = q_vals.gather(1, act_batch)

        # comparing the q values to the values expected using the next states and reward
        #next_state_values = Variable(torch.zeros(32).cuda())
        next_state_values = model(next_obs_batch).data.max(1)[0]*not_done_mask.view(-1, 1)
        expected_state_action_values = (next_state_values * gamma) + rew_batch

        # calculating the q loss and n-step return loss
        #print(state_action_values.shape)
        #print(expected_state_action_values.shape)
        q_loss = F.mse_loss(state_action_values, expected_state_action_values, reduction='mean')

        # calculating the supervised loss
        num_actions = q_vals.size(1)
        margins = (torch.ones(num_actions, num_actions) - torch.eye(num_actions)) * 0.5
        batch_margins = margins[act_batch.data.squeeze().cpu()]
        q_vals = q_vals + Variable(batch_margins).type(dtype)
        supervised_loss = (q_vals.max(1)[0].unsqueeze(1) - state_action_values).pow(2).sum()
        #print("SU_LOSS:",supervised_loss)

        loss = supervised_loss + q_loss

        writer.add_scalar('Loss_supervised', loss, step)

        opter.zero_grad()
        loss.backward()

        for param in model.parameters():
            param.grad.data.clamp(-1, 1)
        opter.step()


    Q, target_Q = create_networks()
    # Construct Optimizers
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)
    opter = optim.Adam(Q.parameters(), lr=lr)
    # Construct the replay buffer
    replay_buffer = PrioritizedReplayBuffer(replay_buffer_size, alpha)

    writer = SummaryWriter()

    experience = recollect_experience(env, q_func)
    for x in experience:
        (last_obs,action,reward,obs,done) = x
        replay_buffer.add(last_obs,action,reward,obs,done)

    for i in range(50000):
        #print(i)
        optimize_dqfd(Q,i,writer)

    target_Q.load_state_dict(Q.state_dict())

    last_obs = env.reset()

    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    for i in range(40000):
        action = random.randrange(num_actions)
        obs, reward, done, _ = env.step(action)
        replay_buffer.add(process_observation(last_obs), action, reward, process_observation(obs), done)
        if done:
            obs = env.reset()
        last_obs = obs
    while not done:
        action = random.randrange(num_actions)
        obs, reward, done, _ = env.step(action)
    last_obs = env.reset()
    for t in count():
        action = epsilon_greedy_action(Q, process_observation(last_obs).transpose(2, 0, 1)).numpy()[0, 0]

        obs, reward, done, _ = env.step(action)

        replay_buffer.add(process_observation(last_obs), action, reward, process_observation(obs), done)

        if done:
            obs = env.reset()
        last_obs = obs

        if (t > 0 and
                t % learning_freq == 0):

            num_param_updates = train_step(Q, target_Q, num_param_updates, t, writer)

            # Periodically update the target network by Q network to target Q network
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())

        mean_episode_reward, best_mean_episode_reward = save_info(LOG_EVERY_N_STEPS, t, mean_episode_reward, best_mean_episode_reward)
        writer.add_scalar('Mean Reward', mean_episode_reward, t)

