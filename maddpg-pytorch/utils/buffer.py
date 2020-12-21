import numpy as np
from torch import Tensor
from torch.autograd import Variable
from heapq import *
import itertools

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]



class PriorityReplayBuffer(object):
    """
    Priority Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_traj, episode_length, num_agents, obs_dims, ac_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.obs_dims = obs_dims
        self.acs_dims = ac_dims
        self.max_traj = max_traj
        self.heap = []
        self.num_traj = 0
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.counter = itertools.count()
        self.observations = np.zeros((max_traj*episode_length, sum(self.obs_dims)), dtype=float)
        self.actions = np.zeros((max_traj*episode_length, sum(self.acs_dims)), dtype=float)

    def __len__(self):
        return self.num_traj

    def push_traj(self, current_episode, total, current_trajectory=None):
        if self.num_traj < self.max_traj:
            heappush(self.heap, [total, self.num_traj, current_trajectory])
            #add elements at self.current_ind
            for i in range(len(current_episode)):
                self.observations[self.num_traj*self.episode_length + i] = (current_episode[i][0]).reshape((1, self.observations.shape[1]))
                self.actions[self.num_traj*self.episode_length + i] = np.hstack(current_episode[i][1])
            self.num_traj = self.num_traj + 1
        else:
            elem = self.heap[0]
            if total > elem[0]:
                heappushpop(self.heap, [total, elem[1], current_trajectory])
                #replace state and actions for elem[0]
                for i in range(len(current_episode)):
                    self.observations[elem[1]*self.episode_length + i] = (current_episode[i][0]).reshape((1, self.observations.shape[1]))
                    self.actions[elem[1]*self.episode_length + i] = np.hstack(current_episode[i][1])

    def push(self, current_episode, total, n_rollout_threads, current_entities, current_trajectory, store_traj):
        if store_traj:
            for i in range(n_rollout_threads):
                self.push_traj(current_episode[i], total[i][0], [current_entities[i], current_trajectory[i]])
        else:
            for i in range(n_rollout_threads):
                self.push_traj(current_episode[i], total[i][0])

    def sample(self, N, to_gpu=False, norm_rews=True):
        #random choice or top n ? self.num_traj*self.episode_length
        elements = np.random.choice(np.arange(self.num_traj*self.episode_length), size=N,
                                replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        return ([cast(self.observations[:, i*self.obs_dims[0]:(i+1)*self.obs_dims[0]][elements]) for i in range(self.num_agents)],
                [cast(self.actions[:, i*self.acs_dims[0]:(i+1)*self.acs_dims[0]][elements]) for i in range(self.num_agents)])
    
    def get_average_return(self):
        return np.mean(np.array(self.heap)[:, 0][:self.num_traj])

    def get_trajectories(self):
        return self.heap