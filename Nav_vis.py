from unityagents import UnityEnvironment
import numpy as np


from dqn_agent_vis import Agent
from collections import deque
import torch
import matplotlib.pyplot as plt

def train(number_of_episodes = 2000, max_timesteps = 1000, e_greedy_init = 1.0, e_greedy_decay=0.9995, e_greedy_min = 0.01):
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = e_greedy_init
    
    for i_episode in range(1, number_of_episodes):
        env_info = env.reset(train_mode=True)[brain_name]# reset the environment
        state = env_info.visual_observations[0]            # get the current state
        state = state.reshape((-1,3,84,84))
        score = 0                                          # initialize the score

        for t in range(max_timesteps):
            action = agent.act(state, eps)                 # select an action
            #print (type(action))
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.visual_observations[0]   # get the next state
            next_state = next_state.reshape((-1,3,84,84))
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
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=8:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break 
    return scores

env = UnityEnvironment(file_name="./VisualBanana_Linux/Banana.x86_64")

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

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
print(env_info.visual_observations)

# examine the state space 
state = env_info.visual_observations[0]
print('States look like:')
#plt.imshow(np.squeeze(state))
#plt.show()
state = state.reshape((-1,3,84,84))
state_size = state.shape
print('States have shape:', state.shape)
print('state_size have:', state_size)
(a,b,c,d)=state_size
agent = Agent(b, action_size, seed=0, network="Convolutional", stepkey="Double")

scores = train()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


