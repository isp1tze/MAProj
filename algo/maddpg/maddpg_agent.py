from algo.maddpg.network import Critic, Actor
import torch
from copy import deepcopy
from torch.optim import Adam
from algo.memory import ReplayMemory, Experience
from algo.random_process import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import os
import torch.nn as nn
import numpy as np
from algo.utils import device
scale_reward = 0.01


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, dim_obs, dim_act, n_agents, args):
        self.args = args
        self.mode = args.mode
        self.actors = []
        self.critics = []
        self.actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
        # self.critic = Critic(n_agents, dim_obs, dim_act)
        self.critics = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.memory = ReplayMemory(args.memory_length)
        self.batch_size = args.batch_size
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = args.episode_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]

        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def load_model(self):
        if self.args.model_episode:
            path_flag = True
            for idx in range(self.n_agents):
                path_flag = path_flag \
                            and (os.path.exists("trained_model/maddpg/actor["+ str(idx) + "]_"
                                                +str(self.args.model_episode)+".pth")) \
                            and (os.path.exists("trained_model/maddpg/critic["+ str(idx) + "]_"
                                                +str(self.args.model_episode)+".pth"))

            if path_flag:
                print("load model!")
                for idx in range(self.n_agents):
                    actor = torch.load("trained_model/maddpg/actor["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                    critic = torch.load("trained_model/maddpg/critic["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                    self.actors[idx].load_state_dict(actor.state_dict())
                    self.critics[idx].load_state_dict(critic.state_dict())

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

    def save_model(self, episode):
        if not os.path.exists("./trained_model/" + str(self.args.algo) + "/"):
            os.mkdir("./trained_model/" + str(self.args.algo) + "/")
        for i in range(self.n_agents):
            torch.save(self.actors[i],
                       'trained_model/maddpg/actor[' + str(i) + ']' + '_' + str(episode) + '.pth')
            torch.save(self.critics[i],
                       'trained_model/maddpg/critic[' + str(i) + ']' + '_' + str(episode) + '.pth')

    def update(self,i_episode):

        self.train_num = i_episode
        if self.train_num <= self.episodes_before_train:
            return None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        for agent in range(self.n_agents):

            non_final_mask = BoolTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).type(FloatTensor)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            non_final_next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)

            self.actor_optimizer[agent].zero_grad()
            self.critic_optimizer[agent].zero_grad()
            self.actors[agent].zero_grad()
            self.critics[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)
            non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i,:]) for i in range(self.n_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())
            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states), # .view(-1, self.n_agents * self.n_states)
                non_final_next_actions.view(-1, self.n_agents * self.n_actions)).squeeze() # .view(-1, self.n_agents * self.n_actions)

            # scale_reward: to scale reward in Q functions
            reward_sum = sum([reward_batch[:,agent_idx] for agent_idx in range(self.n_agents)])
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1)*0.1)# + reward_sum.unsqueeze(1) * 0.1

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            self.critic_optimizer[agent].zero_grad()
            self.actors[agent].zero_grad()
            self.critics[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action).mean()
            # actor_loss += (action_i ** 2).mean() * 1e-3
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)
            self.actor_optimizer[agent].step()
            # self.critic_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.train_num % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return sum(c_loss).item()/self.n_agents, sum(a_loss).item()/self.n_agents

    def choose_action(self, state, noisy=True):
        obs = torch.from_numpy(np.stack(state)).float().to(device)
        actions = torch.zeros(self.n_agents, self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        for i in range(self.n_agents):
            sb = obs[i].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()
            if noisy:
                act += torch.from_numpy(np.random.randn(2) * self.var[i]).type(FloatTensor)

                if self.episode_done > self.episodes_before_train and \
                        self.var[i] > 0.05:
                    self.var[i] *= 0.999998
            act = torch.clamp(act, -1.0, 1.0)

            actions[i, :] = act
        self.steps_done += 1
        return actions.data.cpu().numpy()
