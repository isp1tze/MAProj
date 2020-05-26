import torch
import torch.nn as nn

from algo.utils import weight_init
from algo.utils import fanin_init

HIDDEN_DIM = 300


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, n_agents):
        super(Actor, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n_agents = n_agents

        # input (batch, s_dim) output (batch, 300)

        self.prev_dense = DenseNet(s_dim, HIDDEN_DIM, HIDDEN_DIM // 2, output_activation=None, norm_in=False)
        # input (num_agents, batch, 200) output (num_agents, batch, num_agents * 2)\
        self.comm_net = LSTMNet(HIDDEN_DIM // 2, HIDDEN_DIM // 2, num_layers=1)
        # input (batch, 2) output (batch, a_dim)
        self.post_dense = DenseNet(HIDDEN_DIM + s_dim, HIDDEN_DIM // 2, a_dim, output_activation=nn.Tanh)

    def forward(self, x):
        x_s = x
        x = x.view(-1, self.s_dim)
        x = self.prev_dense(x)
        x = x.reshape(-1, self.n_agents, HIDDEN_DIM // 2)
        x = self.comm_net(x)
        x = torch.cat((x, x_s), dim=-1)
        x = x.reshape(-1, HIDDEN_DIM + self.s_dim)
        x = self.post_dense(x)
        x = x.view(-1, self.n_agents, self.a_dim)
        return x


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, n_agents):
        super(Critic, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n_agents = n_agents

        # input (batch, s_dim) output (batch, 300)
        self.prev_dense = DenseNet((s_dim + a_dim), HIDDEN_DIM, HIDDEN_DIM // 2, output_activation=None, norm_in=False)
        # input (num_agents, batch, 200) output (num_agents, batch, num_agents * 2)\
        self.comm_net = LSTMNet(HIDDEN_DIM // 2, HIDDEN_DIM // 2, num_layers=1)
        # input (batch, 2) output (batch, a_dim)
        self.post_dense = DenseNet(HIDDEN_DIM + s_dim, HIDDEN_DIM // 2, 1, output_activation=None)

    def forward(self, x_n, a_n):
        x = torch.cat((x_n, a_n), dim=-1)
        x = x.view(-1, (self.s_dim + self.a_dim))
        x = self.prev_dense(x)

        x = x.reshape(-1, self.n_agents, HIDDEN_DIM // 2)
        x = self.comm_net(x)
        x = torch.cat((x, x_n), dim=-1)
        x = x.reshape(-1, HIDDEN_DIM + self.s_dim)

        x = self.post_dense(x)
        x = x.view(-1, self.n_agents, 1)
        return x


class DenseNet(nn.Module):
    def __init__(self, s_dim, hidden_dim, a_dim, norm_in=False, hidden_activation=nn.ReLU, output_activation=None):
        super(DenseNet, self).__init__()

        self._norm_in = norm_in

        if self._norm_in:
            self.norm1 = nn.BatchNorm1d(s_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
            self.norm3 = nn.BatchNorm1d(hidden_dim)
            self.norm4 = nn.BatchNorm1d(hidden_dim)

        self.dense1 = nn.Linear(s_dim, hidden_dim)
        self.dense1.weight.data = fanin_init(self.dense1.weight.data.size())
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2.weight.data = fanin_init(self.dense2.weight.data.size())
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3.weight.data.uniform_(-0.003, 0.003)
        self.dense4 = nn.Linear(hidden_dim, a_dim)

        if hidden_activation:
            self.hidden_activation = hidden_activation()
        else:
            self.hidden_activation = lambda x : x

        if output_activation:
            self.output_activation = output_activation()
        else:
            self.output_activation = lambda x : x

    def forward(self, x):
        use_norm = True if (self._norm_in and x.shape[0] != 1) else False

        if use_norm: x = self.norm1(x)
        x = self.hidden_activation(self.dense1(x))
        if use_norm: x = self.norm2(x)
        x = self.hidden_activation(self.dense2(x))
        if use_norm: x = self.norm3(x)
        x = self.hidden_activation(self.dense3(x))
        if use_norm: x = self.norm4(x)
        x = self.output_activation(self.dense4(x))
        return x


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1,
                 bias=True,
                 batch_fisrt=True,
                 bidirectional=True):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_fisrt,
            bidirectional=bidirectional
        )

    def forward(self, input, wh=None, wc=None):
        output, (hidden, cell) = self.lstm(input)
        return output
