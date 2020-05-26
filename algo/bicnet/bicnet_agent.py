import torch, os
import numpy as np, random

from algo.bicnet.network import Actor, Critic
from algo.random_process import OrnsteinUhlenbeckProcess
from algo.utils import soft_update, hard_update, device

class BiCNet():

    def __init__(self, s_dim, a_dim, n_agents, args):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.config = args
        self.n_agents = n_agents
        self.device = device
        # Networks
        self.actor = Actor(s_dim, a_dim, n_agents).to(device)
        self.actor_target = Actor(s_dim, a_dim, n_agents).to(device)
        self.critic = Critic(s_dim, a_dim, n_agents).to(device)
        self.critic_target = Critic(s_dim, a_dim, n_agents).to(device)


        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.c_lr)

        hard_update(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)

        self.random_process = OrnsteinUhlenbeckProcess(size=self.a_dim,
                                                       theta=self.config.ou_theta,
                                                       mu=self.config.ou_mu,
                                                       sigma=self.config.ou_sigma)
        self.replay_buffer = list()
        self.epsilon = 1.
        self.depsilon = self.epsilon / self.config.epsilon_decay
        self.batch_size = self.config.batch_size

        self.c_loss = None
        self.a_loss = None
        self.action_log = list()
        self.train_num = 0
        self.var = [1.0 for i in range(n_agents)]

    def load_model(self):
        model_actor_path = "./trained_model/" + str(self.config.algo) + "/actor_" + str(self.config.model_episode) + ".pth"
        model_critic_path = "./trained_model/" + str(self.config.algo) + "/critic_" + str(self.config.model_episode) + ".pth"
        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            print("load model!")
            actor = torch.load(model_actor_path)
            critic = torch.load(model_critic_path)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)

    def save_model(self, episode):
        if not os.path.exists("./trained_model/" + str(self.config.algo) + "/"):
            os.mkdir("./trained_model/" + str(self.config.algo) + "/")
        torch.save(self.actor.state_dict(),
                   "./trained_model/" + str(self.config.algo) + "/actor_" + str(episode) + ".pth"),
        torch.save(self.critic.state_dict(),
                   "./trained_model/" + str(self.config.algo) + "/critic_" + str(episode) + ".pth"),


    def choose_action(self, obs, noisy=True):
        obs = torch.Tensor([obs]).to(self.device)

        action = self.actor(obs).cpu().detach().numpy()[0]
        self.action_log.append(action)

        if noisy:
            for agent_idx in range(self.n_agents):
                action[agent_idx] += np.random.randn(2) * self.var[agent_idx]

                if self.var[agent_idx] > 0.05:
                    self.var[agent_idx] *= 0.999998
        action = np.clip(action, -1., 1.)

        return action

    def reset(self):
        self.random_process.reset_states()
        self.action_log.clear()

    def prep_train(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def prep_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def random_action(self):
        return np.random.uniform(low=-1, high=1, size=(self.n_agents, 2))

    def memory(self, s, a, r, s_, done):
        self.replay_buffer.append((s, a, r, s_, done))

        if len(self.replay_buffer) >= self.config.memory_length:
            self.replay_buffer.pop(0)

    def get_batches(self):
        experiences = random.sample(self.replay_buffer, self.config.batch_size)

        state_batches = np.array([_[0] for _ in experiences])
        action_batches = np.array([_[1] for _ in experiences])
        reward_batches = np.array([_[2] for _ in experiences])
        next_state_batches = np.array([_[3] for _ in experiences])
        done_batches = np.array([_[4] for _ in experiences])


        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def update(self,i_episode):

        if len(self.replay_buffer) < self.batch_size:
            return None, None

        state_batches, action_batches, reward_batches, next_state_batches, done_batches = self.get_batches()
        state_batches = torch.Tensor(state_batches).to(self.device)
        action_batches = torch.Tensor(action_batches).to(self.device)
        reward_batches = torch.Tensor(reward_batches).reshape(self.config.batch_size, self.n_agents, 1).to(self.device)
        next_state_batches = torch.Tensor(next_state_batches).to(self.device)
        done_batches = torch.Tensor((done_batches == False) * 1).reshape(self.config.batch_size, self.n_agents, 1).to(self.device)

        target_next_actions = self.actor_target(next_state_batches)
        target_next_q = self.critic_target(next_state_batches, target_next_actions)
        main_q = self.critic(state_batches, action_batches)

        # Critic Loss
        self.critic.zero_grad()
        self.critic_optimizer.zero_grad()
        baselines = reward_batches + done_batches * self.config.gamma * target_next_q
        loss_critic = torch.nn.MSELoss()(main_q, baselines.detach())
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Actor Loss
        self.actor.zero_grad()
        self.actor_optimizer.zero_grad()
        self.critic.zero_grad()
        self.critic_optimizer.zero_grad()
        clear_action_batches = self.actor(state_batches)
        loss_actor = -self.critic(state_batches, clear_action_batches).mean()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        self.c_loss = loss_critic.item()
        self.a_loss = loss_actor.item()

        self.train_num = i_episode

        if self.train_num % 100 == 0:
            soft_update(self.actor, self.actor_target, self.config.tau)
            soft_update(self.critic, self.critic_target, self.config.tau)

        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss

    def get_action_std(self):
        return np.array(self.action_log).std(axis=-1).mean()
