import sys
import pickle
import os
import numpy as np
from collections import namedtuple
from itertools import count
import random

import torch
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}

def dqn_learn(
	env,
	q_func,
	optimizer_spec,
    exploration,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=500,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000
	):

	img_h, img_w, img_c = env.observation_space.shape
	input_arg = frame_history_len * img_c
	num_actions = 11


	def epsilon_greedy_action(model, obs, t):
		sample = random.random()
		eps = exploration.value(t)
		if sample > eps:

			obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)/255.0

			return model(Variable(obs)).data.max(1)[1].view(-1,1).cpu()

		else:
			return torch.IntTensor([[random.randrange(num_actions)]]).cpu()

	Q = q_func(input_arg,num_actions).type(dtype)
	target_Q = q_func(input_arg,num_actions).type(dtype)

	if os.path.isfile('mario_Q_params.pkl'):
		print('Load Q parametets ...')
		Q.load_state_dict(torch.load('mario_Q_params.pkl'))
        
	if os.path.isfile('mario_target_Q_params.pkl'):
		print('Load target Q parameters ...')
		target_Q.load_state_dict(torch.load('mario_target_Q_params.pkl'))

	# Construct Optimizer
	optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

	# Construct the replay buffer
	replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

	last_obs = env.reset()

	num_param_updates = 0
	mean_episode_reward = -float('nan')
	best_mean_episode_reward = -float('inf')
	last_obs = env.reset()
	LOG_EVERY_N_STEPS  = 10000

	for t in count():

		last_idx = replay_buffer.store_frame(last_obs)


		recent_obs = replay_buffer.encode_recent_observation()

		if t > learning_starts:
			#print("epsilon Greedy action")
			action = epsilon_greedy_action(Q,recent_obs,t).numpy()[0,0]
			##print(action)
		else:
			action = random.randrange(num_actions)

		#print("to do action... ")
		#print(action)

		obs, reward, done, _ = env.step(action)

		reward = max(-1.0, min(reward, 1.0))

		replay_buffer.store_effect(last_idx, action, reward, done)

		if done:
			obs = env.reset()
		last_obs = obs

		if (t > learning_starts and
			t % learning_freq == 0 and
			replay_buffer.can_sample(batch_size)):
			##print("Learning in t: %d",t)
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
			obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
            # Convert numpy nd_array to torch variables for calculation
			obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
			act_batch = Variable(torch.from_numpy(act_batch).long())
			rew_batch = Variable(torch.from_numpy(rew_batch))
			next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
			not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

			if USE_CUDA:
				act_batch = act_batch.cuda()
				rew_batch = rew_batch.cuda()

            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # We choose Q based on action taken.
			current_Q_values = Q(obs_batch).gather(1, act_batch.view(-1, 1))
            
			"""
            # DQN
            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            next_max_q = target_Q(next_obs_batch).detach().max(1)[0].view(-1, 1)
            next_Q_values = not_done_mask.view(-1, 1) * next_max_q
            """
			next_argmax_action = Q(next_obs_batch).max(1)[1].view(-1, 1)
			next_q = target_Q(next_obs_batch).detach().gather(1, next_argmax_action)
			next_Q_values = not_done_mask.view(-1, 1) * next_q 
            # Compute the target of the current Q values
			target_Q_values = rew_batch.view(-1, 1) + (gamma * next_Q_values)
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
			loss = F.smooth_l1_loss(current_Q_values, target_Q_values)
			optimizer.zero_grad()
			loss.backward()
			for param in Q.parameters():
				param.grad.data.clamp(-1, 1)
            # Perfom the update
			optimizer.step()
			num_param_updates += 1

            # Periodically update the target network by Q network to target Q network
			##print("num updates %d",num_param_updates)
			if num_param_updates % target_update_freq == 0:
				target_Q.load_state_dict(Q.state_dict())

		episode_rewards = env.get_episode_rewards()
		if len(episode_rewards) > 0:
			mean_episode_reward = np.mean(episode_rewards[-100:])
		if len(episode_rewards) > 100:
			best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

		Statistic["mean_episode_rewards"].append(mean_episode_reward)
		Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

		if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
			print("Timestep %d" % (t,))
			print("mean reward (100 episodes) %f" % mean_episode_reward)
			print("best mean reward %f" % best_mean_episode_reward)
			print("episodes %d" % len(episode_rewards))
			print("exploration %f" % exploration.value(t))
			torch.save(Q.state_dict(), 'mario_Q_params.pkl')
			torch.save(target_Q.state_dict(), 'mario_target_Q_params.pkl')
			sys.stdout.flush()

            # Dump statistics to pickle
			with open('statistics.pkl', 'wb') as f:
				pickle.dump(Statistic, f)
				print("Saved to %s" % 'statistics.pkl')

