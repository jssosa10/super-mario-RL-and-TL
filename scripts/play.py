import os
from itertools import count

import torch
import torch.autograd as autograd


USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


def dqn_play(
    env,
        q_func):

    img_h, img_w, img_c = env.observation_space.shape
    input_arg = img_c
    num_actions = env.action_space.n

    def load_model(model):
        if os.path.isfile('mario_Q_params.pkl'):
            print('Load Q parametets ...')
            model.load_state_dict(torch.load('mario_Q_params.pkl'))
        return model

    def get_action(model, obs):
        obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)/255.0
        return model(Variable(obs)).data.max(1)[1].view(-1, 1).cpu()

    Q = q_func(input_arg, num_actions).type(dtype)
    Q = load_model(Q)

    obs = env.reset()

    for t in count():

        action = get_action(Q, obs.transpose(2, 0, 1)).numpy()[0, 0]

        obs, reward, done, _ = env.step(action)

        if done:
            obs = env.reset()
