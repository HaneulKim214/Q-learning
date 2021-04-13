from collections import defaultdict
import numpy as np

class Agent(object):

    def __init__(self, env, learning_rate=0.01, discount_value=0.9,
                 epsilon=0.9, epsilon_min=0.1, epsilon_decay=0.95):
        self.env = env
        self.lr = learning_rate
        self.discount_value = discount_value
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_action_space = env.action_space.n

        self.q_table = defaultdict(lambda: np.zeros(self.num_action_space))

    def choose_action(self, state):
        """choose action at current state by exploring or exploit depending on epsilon value"""
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.num_action_space)
        else:
            q_vals = self.q_table[state]
            perm_actions = np.random.permutation(self.num_action_space)
            q_vals = [q_vals[a] for a in perm_actions] # randomizing q_vals -> why?
            perm_q_argmax = np.argmax(q_vals)
            action = perm_actions[perm_q_argmax]
        return action

    def learn(self, transition):
        """update action-value function"""
        s, a, r, next_s, done = transition
        q_val = self.q_table[s][a]
        if done:
            q_target = r
        else:
            q_target = r + self.discount_value * np.max(self.q_table[next_s])
        self.q_table[s][a] += self.lr * (q_target - q_val)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay