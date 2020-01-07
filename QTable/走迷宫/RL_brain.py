import numpy as np
import pandas as pd


class QLearningTable:
    """
    q_table 的 index 由动作组成
    """
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        # 初始动作；学习率；反馈比率；非随机比率；QTable初始化
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # (1 - epsilon) 几率随机走，epsilon 几率通过 QLearning 选择行为
        if np.random.uniform() < self.epsilon:
            # 行为位置，列为动作；选择当前位置所有动作；值相等，随机选择
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, action, reward, s_):
        # 检查是否存在这个位置的动作，不存在则添加
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, action]
        if s_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = reward
        # q_target 是回馈加上下一步的得分 * gamma；如果 q_target 高于当前，说明下一步值得走
        self.q_table.loc[s, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        # 如果状态不存在，添加该行
        print(self.q_table.index)

        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )