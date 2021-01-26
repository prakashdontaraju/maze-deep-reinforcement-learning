import matplotlib.pyplot as plt
import numpy as np

def show_loss_statistics(episode_statistics,loss_statistics):
    
    episodes = episode_statistics
    # print(episodes)
    x_label = 'Episodes'
    y_label='Loss'

    # loss = np.array([0.08525, 0.075632, 0.045242, 0026295.])
    # loss = [np.random.uniform(0.0001, 0.1) for reward in range(1, 22)]
    loss = loss_statistics
    # print(loss)

    plt.figure(1)
    title = 'Loss vs Episodes'
    plt.plot(episodes, loss, linewidth=3.0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def show_reward_statistics(episode_statistics, rewards_statistics) :
    
    episodes = episode_statistics
    # print(episodes)
    x_label = 'Episodes'
    y_label='Rewards'

    # rewards = np.array([-103, -116, 97, -110])
    # rewards = [np.random.uniform(80.1, 100.0) for reward in range(1, 22)]
    rewards = rewards_statistics
    # print(rewards)

    plt.figure(1)
    title = 'Rewards vs Episodes'
    plt.plot(episodes, rewards, linewidth=3.0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()