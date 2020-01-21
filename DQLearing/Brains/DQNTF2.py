import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class DQN(tf.keras.Model):
    def __init__(
            self,
            n_actions,
            n_features,
            n_hidden
    ):
        super(DQN, self).__init__(name='DQN')
        self.net = tf.keras.Sequential(
            [layers.Dense(n_features, activation=tf.keras.activations.relu),
            layers.Dense(n_hidden, activation=tf.keras.activations.relu),
            layers.Dense(units=n_actions, activation=tf.keras.activations.softmax)]
        )

    def call(self, inputs, training=None, mask=None):
        return self.net(inputs)

class DeepQN():
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
    ):
        self.memory_counter = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.reward_decay = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.memory = np.zeros((memory_size, n_features*2 + 2))
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.cost_his = []
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        self.eval_net = DQN(n_actions, n_features, n_hidden=10)
        self.eval_net.compile(optimizer='rmsprop',loss=tf.keras.losses.mean_squared_error)
        self.target_net = DQN(n_actions, n_features, n_hidden=10)
        self.target_net.compile(optimizer=tf.keras.optimizers.RMSprop(),
                                loss=tf.keras.losses.mean_squared_error)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            print('predict')
            action_value = self.eval_net.predict(observation)
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            print("replace")
            self.eval_net.save_weights('./tmp_model')
            self.target_net.load_weights('./tmp_model')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index,:]
        q_next = self.target_net.predict(batch_memory[:, -self.n_features])
        q_target = self.eval_net.predict(batch_memory[:, self.n_features])

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.reward_decay * np.max(q_next, axis=1)
        self.eval_net.fit(batch_memory[:, self.n_features], q_target)

        cost = self.eval_net.total_loss

        self.cost_his.append(cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
