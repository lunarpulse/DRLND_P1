from unityagents import UnityEnvironment
import numpy as np


from dqn_PER import Agent
from collections import deque
import torch
import matplotlib.pyplot as plt


def train(number_of_episodes = 2000, max_timesteps = 1000, e_greedy_start = 1.0, e_greedy_decay=0.985, e_greedy_min = 0.008):
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = e_greedy_start
    
    for i_episode in range(1, number_of_episodes):
        env_info = env.reset(train_mode=True)[brain_name]# reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score

        for t in range(max_timesteps):
            action = agent.act(state, eps)                 # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(e_greedy_min, e_greedy_decay*eps) # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f} eps: {:.2f}'.format(i_episode, np.mean(scores_window), eps), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} eps: {:.2f}'.format(i_episode, np.mean(scores_window), eps))
        if np.mean(scores_window)>=13:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break 
    return scores

env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

agent = Agent(state_size, action_size, seed=0, network="Dueling", stepkey="Double")

scores = train()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episodes')
plt.show()

