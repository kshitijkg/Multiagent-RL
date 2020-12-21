import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork, MLPNetwork_Disc
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax, hard_update
from utils.agents import DDPGAgent
import numpy as np
from itertools import permutations
MSELoss = torch.nn.MSELoss()
from utils.misc import hard_update

from torch.autograd import Variable
from torch.optim import Adam

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                discrete_action=False, stochastic = False,
                commonCritic = False, gasil = False, dlr = 0.0003, lambda_disc = 0.5, batch_size_disc = 512, dynamic = False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]

        for i in self.agents:
            i.target_policy.requires_grad = False
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.dlr = dlr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.disc_dev = 'cpu'
        self.niter = 0
        self.stochastic = stochastic
        self.commonCritic = commonCritic
        self.gasil = gasil
        self.lambda_disc = lambda_disc
        self.batch_size_disc = batch_size_disc
        self.dynamic = dynamic
        num_in_critic = self.agent_init_params[0]['num_in_critic']
        self.cuda = True if torch.cuda.is_available() else False 
        if self.commonCritic:
            
            #num_in_discriminator = self.agent_init_params[0]['num_in_pol'] + self.agent_init_params[0]['num_out_pol']
            #This can be changed and looked at 
            
            self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
            self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
            hard_update(self.target_critic, self.critic)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if self.gasil:
            self.discriminator = MLPNetwork_Disc(num_in_critic, 1,
                                 hidden_dim=hidden_dim, norm_in=False,
                                 constrain_out=False, discrete_action=False)
            self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=dlr)
    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]
    
    def permute(self, matrix, matrix_orig, permute_mat):
        stacked = matrix.t().reshape(len(matrix_orig), matrix.size()[0]*matrix_orig[0].size()[1]).t()
        return stacked[:, permute_mat].t().reshape(matrix.size()[1], matrix.size()[0]).t()

    def update(self, sample, agent_i, parallel=False, logger=None, num_AC_permutations = 4):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        rews = [a.view(len(a), 1) for a in rews]
        dones = [a.view(len(a), 1) for a in dones]
        curr_agent = self.agents[agent_i]
        if self.commonCritic:
            current_critic = self.critic
            current_critic_optimiser = self.critic_optimizer
            current_target_critic = self.target_critic
        else:
            current_critic = curr_agent.critic
            current_critic_optimiser = curr_agent.critic_optimizer
            current_target_critic =  curr_agent.target_critic

        if self.alg_types[agent_i] == 'MADDPG':
            if self.discrete_action: # one-hot encode action
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs)]
            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
            obs_cat, acs_cat, rews_cat = torch.cat(obs, dim=1), torch.cat(acs, dim=1), torch.cat(rews, dim=1)
            next_obs_cat, dones_cat = torch.cat(next_obs, dim=1), torch.cat(dones, dim=1)
            all_trgt_acs_cat = torch.cat(all_trgt_acs, dim=1)
            vf_loss_total = 0

            for i in range(0, num_AC_permutations):
                current_critic_optimiser.zero_grad()
                perm_mat = torch.randperm(len(obs))

                all_trgt_acs_cat = self.permute(all_trgt_acs_cat, all_trgt_acs, perm_mat)
                obs_cat = self.permute(obs_cat, obs, perm_mat)
                acs_cat = self.permute(acs_cat, acs, perm_mat)
                rews_cat = self.permute(rews_cat, rews, perm_mat)
                next_obs_cat = self.permute(next_obs_cat, next_obs, perm_mat)
                dones_cat = self.permute(dones_cat, dones, perm_mat)

                trgt_vf_in = torch.cat((next_obs_cat, all_trgt_acs_cat), dim=1)
                target_value = (rews_cat[:, agent_i*rews[0].size()[1]:(agent_i+1)*rews[0].size()[1]].view(-1, 1) + self.gamma *
                                    current_target_critic(trgt_vf_in))

                vf_in = torch.cat((obs_cat, acs_cat), dim=1)

                actual_value = current_critic(vf_in)
                vf_loss = MSELoss(actual_value, target_value.detach())
                if self.stochastic:
                    vf_loss.backward()
                    if parallel:
                        average_gradients(current_critic)
                    torch.nn.utils.clip_grad_norm(current_critic.parameters(), 0.5)
                    current_critic_optimiser.step()
                else:
                    vf_loss_total = vf_loss_total + vf_loss
            
            if not self.stochastic:
                vf_loss_total = vf_loss_total/num_AC_permutations
                vf_loss.backward()
                if parallel:
                    average_gradients(current_critic)
                torch.nn.utils.clip_grad_norm(current_critic.parameters(), 0.5)
                current_critic_optimiser.step()            
        else: 
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)

            target_value = (rews[agent_i].view(-1, 1) + self.gamma * current_target_critic(trgt_vf_in))
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
            actual_value = current_critic(vf_in)
            vf_loss = MSELoss(actual_value, target_value.detach())
            vf_loss.backward()
            if parallel:
                average_gradients(current_critic)
            torch.nn.utils.clip_grad_norm(current_critic.parameters(), 0.5)
            current_critic_optimiser.step()

        for agent_i in range(self.nagents):
            curr_agent.policy_optimizer.zero_grad()

            if self.discrete_action:
                # Forward pass as if onehot (hard=True) but backprop through a differentiable
                # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
                # through discrete categorical samples, but I'm not sure if that is
                # correct since it removes the assumption of a deterministic policy for
                # DDPG. Regardless, discrete policies don't seem to learn properly without it.
                curr_pol_out = curr_agent.policy(obs[agent_i])
                #curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
                curr_pol_vf_in = gumbel_softmax(curr_pol_out)
            else:
                curr_pol_out = curr_agent.policy(obs[agent_i])
                curr_pol_vf_in = curr_pol_out
            
            if self.alg_types[agent_i] == 'MADDPG':
                all_pol_acs = []
                for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                    if i == agent_i:
                        all_pol_acs.append(curr_pol_vf_in)
                    elif self.discrete_action:
                        all_pol_acs.append(onehot_from_logits(curr_agent.policy(ob)))
                    else:
                        all_pol_acs.append(curr_agent.policy(ob))
                vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
            else:  # DDPG
                vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                                dim=1)
            pol_loss = -current_critic(vf_in).mean()
            #pol_loss += (curr_pol_out**2).mean() * 1e-3
            pol_loss.backward()
            if parallel:
                average_gradients(curr_agent.policy)
            torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            if logger is not None:
                logger.add_scalars('agent%i/losses' % agent_i,
                                {'vf_loss': vf_loss,
                                    'pol_loss': pol_loss},
                                self.niter)

    def ones_target(self, size):
        '''
        Tensor containing ones, with shape = size
        '''
        data = Variable(torch.ones(size, 1))
        if self.cuda:
            data = data.cuda()
        return data

    def zeros_target(self, size):
        '''
        Tensor containing zeros, with shape = size
        '''
        data = Variable(torch.zeros(size, 1))
        if self.cuda:
            data = data.cuda()
        return data

    def gasil_disc_update(self, sample_normal, sample_expert, agent_i, parallel=False, logger=None, num_disc_permutations = 4):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            sample_expert: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled from
                    the expert replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        #Update Discriminator
        curr_agent = self.agents[agent_i] 
        obs, acs, rews, next_obs, dones = sample_normal
        rews = [a.view(len(a), 1) for a in rews]
        dones = [a.view(len(a), 1) for a in dones]
        

        obs_exp, acs_exp = sample_expert
        obs_exp_cat, acs_exp_cat = torch.cat(obs_exp, dim=1), torch.cat(acs_exp, dim=1)

        obs_cat, acs_cat, rews_cat = torch.cat(obs, dim=1), torch.cat(acs, dim=1), torch.cat(rews, dim=1)
        next_obs_cat, dones_cat = torch.cat(next_obs, dim=1), torch.cat(dones, dim=1)

        for i in range(0, num_disc_permutations):
            self.discriminator_optimizer.zero_grad()
            perm_mat = torch.randperm(len(obs))

            obs_cat = self.permute(obs_cat, obs, perm_mat)
            acs_cat = self.permute(acs_cat, acs, perm_mat)
            obs_exp_cat = self.permute(obs_exp_cat, obs_exp, perm_mat)
            acs_exp_cat = self.permute(acs_exp_cat, acs_exp, perm_mat)


            vf_in = torch.cat((obs_cat, acs_cat), dim=1)
            vf_in_exp = torch.cat((obs_exp_cat, acs_exp_cat), dim=1)
            
            loss = torch.nn.BCEWithLogitsLoss()
            N = vf_in_exp.size(0)
            prediction_real = self.discriminator(vf_in_exp)
            error_real  = loss(prediction_real, self.ones_target(N))

            N = vf_in.size(0)
            prediction_fake = self.discriminator(vf_in)
            error_fake = loss(prediction_fake, self.zeros_target(N))
            
            total_error = error_real + error_fake
            total_error.backward()
            if logger is not None:
                logger.add_scalars('Discriminator',
                                {'error_real': error_real,
                                    'error_fake': error_fake, 
                                    'total_error' : total_error},
                                self.niter)
            torch.nn.utils.clip_grad_norm(self.discriminator.parameters(), 0.5)
            self.discriminator_optimizer.step()
        

    def gasil_AC_update(self, sample_normal, agent_i, episode_num, parallel=False, logger=None, rew_shape=0, num_AC_permutations = 4):
        #Calculate imitation reward 
        obs, acs, rews, next_obs, dones = sample_normal
        rews = [a.view(len(a), 1) for a in rews]
        dones = [a.view(len(a), 1) for a in dones]
        obs_cat, acs_cat, rews_cat = torch.cat(obs, dim=1), torch.cat(acs, dim=1), torch.cat(rews, dim=1)
        next_obs_cat, dones_cat = torch.cat(next_obs, dim=1), torch.cat(dones, dim=1)
        vf_in = torch.cat((obs_cat, acs_cat), dim=1)
        disc_out_without_sigmoid = self.discriminator(vf_in).detach()
        disc_out = F.sigmoid(disc_out_without_sigmoid)
        if rew_shape == 0:
            rimit = torch.log(disc_out + 1e-3) - torch.log(1 - disc_out + 1e-3)
        elif rew_shape == 1:
            rimit = torch.log(disc_out + 1e-3)
        else:
            rimit = -1*torch.log(1 - disc_out + 1e-3)
        if sum(rimit) == float('-inf') or sum(rimit) != sum(rimit):
            exit()

        #Calculate reshaped rewards
        if self.dynamic:
            rimit = rews[0] + (1 - (1/episode_num + 1))*rimit
        else:
            rimit = self.lambda_disc*rews[0] + rimit
        #Update Real rewards (Do we update only in function or permanently)
        new_sample = (obs, acs, [rimit for i in range(self.nagents)], next_obs, dones)
        
        #Update policy and critic 
        self.update(new_sample, agent_i, logger=logger, num_AC_permutations = num_AC_permutations)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        if self.commonCritic:
            soft_update(self.target_critic, self.critic, self.tau)
        for a_i in range(len(self.agents)):
            a = self.agents[a_i]
            if not self.commonCritic:
                soft_update(a.target_critic, a.critic, self.tau)
            if a_i == 0:
                soft_update(a.target_policy, a.policy, self.tau)
            else:
                hard_update(a.policy, self.agents[0].policy)
                soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        #Fix add train for everything
        if self.commonCritic:
            self.critic.train()
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            if self.commonCritic:
                self.critic = fn(self.critic)
            else:
                for a in self.agents:
                    a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            if self.commonCritic:
                self.target_critic = fn(self.target_critic)
            else:
                for a in self.agents:
                    a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device
        if self.gasil:
            if not self.disc_dev == device:
                self.discriminator = fn(self.discriminator)
                self.disc_dev = device

    def prep_rollouts(self, device='cpu'):
        #fix: add eval for everything 
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                    gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, stochastic = False,
                    commonCritic = False, gasil = False, dlr = 0.0003, lambda_disc = 0.5, batch_size_disc = 512, dynamic = False):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action, 
                     'stochastic' : stochastic, 
                     'commonCritic' : commonCritic, 
                     'gasil' : gasil, 
                     'dlr' : dlr, 
                     'lambda_disc' : lambda_disc, 
                     'batch_size_disc' : batch_size_disc, 
                     'dynamic' : dynamic}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=torch.device('cpu'))
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance