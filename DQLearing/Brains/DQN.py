import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import Variable


class DeepQ(nn.Module):
    def __init__(
            self,
            n_actions,
            n_features,
            hidden=10
    ):
        super(DeepQ, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class DeepQNet():
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.memory_counter = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.cost_his = []
        hidden = 10
        self.eval_net, self.target_net = DeepQ(n_actions, n_features, hidden), DeepQ(n_actions, n_features, hidden)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size

        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.Tensor(observation[np.newaxis, :])

        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(observation)
            action = np.argmax(actions_value.detach().numpy())
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
        # 随机选取读取记忆的坐标
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # q_next 用旧网络预测，q_eval，q_target 用新网络；同时预测记忆中的情形
        q_next, q_eval = self.target_net(torch.Tensor(batch_memory[:, -self.n_features:])).detach().numpy(), \
                         self.eval_net(torch.Tensor(batch_memory[:, :self.n_features])).detach().numpy()
        q_target = q_eval#np.zeros((len(sample_index), self.n_actions))
        # ；动作在 n_features 处；reward 在动作后
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        # 旧网络中的最高价值与新网络对应动作相减得到损失
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        loss = self.loss_fn(self.eval_net(torch.Tensor(batch_memory[:, :self.n_features])), torch.Tensor(q_target))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        cost = loss.item()

        self.cost_his.append(cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
