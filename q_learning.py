from collections import namedtuple
import gym
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from gridworld_env import GridWorldEnv
from agent import Agent

np.random.seed(1)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def q_learning(agent, env, num_ep):
    history = []
    for ep_i in range(num_ep):
        state = env.reset()
        final_reward, n_moves = 0.0, 0
        env.render()
        while True:
            action = agent.choose_action(state)
            next_s, reward, done, info = env.step(action)
            agent.learn(Transition(state, action, reward, next_s, done))
            # env.render(mode='human', done=done)
            env.render()
            state = next_s
            n_moves += 1
            if done:
                break
            final_reward += reward
        history.append((n_moves, final_reward))
        print(f'1Episode {ep_i}: Reward {round(final_reward,2)} #Moves {n_moves}')
    return history

def plot_learning_history(history, env_name):
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Number of Moves", "Final Rewards"))
    for i in range(len(history[0])):
        fig.add_trace(
            go.Scatter(
                x=[j for j in range(len(history))],
                y=[h[i] for h in history]),
            row=i+1, col=1
        )
    fig.update_yaxes(title_text="# moves",row=1,col=1)
    fig.update_yaxes(title_text="Final Rewards", row=2, col=1)
    fig.update_layout(title_text=env_name)
    # fig.show()
    fig.write_html(f"Q-learning_history graph_{env_name}.html")
    # fig.write_image("learning_history_static_graph.png")

if __name__ == '__main__':
    # env_name = "GridWorld"
    # env = GridWorldEnv(num_rows=5, num_cols=6)
    env_name = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    agent = Agent(env)
    learning_history = q_learning(agent, env, num_ep=50)
    env.close()
    plot_learning_history(learning_history, env_name)
