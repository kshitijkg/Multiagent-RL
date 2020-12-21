import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer, PriorityReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
import pickle

USE_CUDA = torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action, benchmark):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, benchmark, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    print(str(log_dir))
    logger = SummaryWriter(str(log_dir))
    #logger = None

    f = open(run_dir / "hyperparametrs.txt","w+")
    f.write(str(config))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action, config.benchmark)
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim, 
                                  stochastic = config.stochastic, 
                                  commonCritic = config.commonCritic, gasil = config.gasil, dlr = config.dlr, lambda_disc = config.lambda_disc,
                                  batch_size_disc = config.batch_size_disc, dynamic=config.dynamic)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    expert_replay_buffer = PriorityReplayBuffer(config.expert_buffer_length, config.episode_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    agent_info = [[[] for i in range(config.n_rollout_threads)]]
    reward_info = []
    total_returns = []
    eval_trajectories = []
    expert_average_returns = []
    trajectories = []
    durations = []
    start_time = time.time()
    expert_trajectories = []
    evaluation_rewards = []
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        if ep_i%100 == 0:
            mins = (time.time() - start_time)/60
            durations.append(mins)
            print(mins, "minutes")
            start_time = time.time()

        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()
        current_episode = [[] for i in range(config.n_rollout_threads)]
        current_trajectory = [[] for i in range(config.n_rollout_threads)]
        current_entities = []
        total_dense = None
        if config.store_traj:
            cur_state_ent = env.getStateEntities()
            for i in range(config.n_rollout_threads):
                current_entities.append(cur_state_ent[i])
           
            cur_state = env.getState()
            for i in range(config.n_rollout_threads):
                current_trajectory[i].append(cur_state[i])
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            if config.store_traj:
                cur_state = env.getState()
                for i in range(config.n_rollout_threads):
                    current_trajectory[i].append(cur_state[i])

            for i in range(config.n_rollout_threads):
                current_episode[i].append([obs[i], actions[i]])
            
            if config.benchmark:
                #Fix this
                for i, info in enumerate(infos):
                    agent_info[-1][i].append(info['n'])

            if et_i == 0:
                total_dense = rewards
            else:
                total_dense = total_dense + rewards

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads and
                ((expert_replay_buffer.num_traj*config.episode_length >= config.batch_size_disc) == (maddpg.gasil))):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                if maddpg.gasil:
                    for update_i in range(config.num_disc_updates):
                        sample_normal = replay_buffer.sample(config.batch_size,to_gpu=USE_CUDA, norm_rews = False)
                        
                        sample_expert = expert_replay_buffer.sample(config.batch_size_disc,
                                                    to_gpu=USE_CUDA)
                        maddpg.gasil_disc_update(sample_normal, sample_expert, 0, logger=logger, num_disc_permutations = config.num_disc_permutations)

                    for update_i in range(config.num_AC_updates):
                        sample_normal = replay_buffer.sample(config.batch_size,to_gpu=USE_CUDA, norm_rews = False)
                        maddpg.gasil_AC_update(sample_normal, 0, episode_num = ep_i, logger=logger, num_AC_permutations = config.num_AC_permutations) 
                else:
                    for update_i in range(config.num_AC_updates):
                        sample_normal = replay_buffer.sample(config.batch_size,to_gpu=USE_CUDA, norm_rews = False)
                        maddpg.update(sample_normal, 0, logger=logger, num_AC_permutations = config.num_AC_permutations)
                maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        total_returns.append(total_dense)
        if maddpg.gasil:
            expert_replay_buffer.push(current_episode, total_dense, config.n_rollout_threads, current_entities, current_trajectory, config.store_traj)
            expert_average_returns.append(expert_replay_buffer.get_average_return())
        
        if config.store_traj:
            for i in range(config.n_rollout_threads):
                trajectories.append([current_entities[i], current_trajectory[i]])

        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
           logger.add_scalars('agent%i/rew' % a_i,
                              {'mean_episode_rewards': a_ep_rew},
                              ep_i)
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        
        #save mean episode rewards
        #save benchmarking data
        agent_info.append([[] for i in range(config.n_rollout_threads)])
        reward_info.append(ep_rews)
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')
            #save the trajectories in the expert replay buffer 
            trajec = expert_replay_buffer.get_trajectories()
            if config.store_traj:
                expert_trajectories.append(trajec)
        
        if ep_i % config.eval_interval < config.n_rollout_threads:
            current_eval = []
            current_trajectories = []

            for ep_i_eval in range(0, config.n_eval_episodes, config.n_rollout_threads):
                obs = env.reset()
                total_eval = None
                maddpg.prep_rollouts(device='cpu')

                if config.store_traj:
                    current_trajectory = [[] for i in range(config.n_rollout_threads)]
                    current_entities = []
                    cur_state_ent = env.getStateEntities()
                    for i in range(config.n_rollout_threads):
                        current_entities.append(cur_state_ent[i])

                    cur_state = env.getState()
                    for i in range(config.n_rollout_threads):
                        current_trajectory[i].append(cur_state[i])

                for et_i in range(config.episode_length):
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                        requires_grad=False)
                                for i in range(maddpg.nagents)]
                    torch_agent_actions = maddpg.step(torch_obs, explore=False)
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                    actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                    next_obs, rewards, dones, infos = env.step(actions)
                    if config.store_traj:
                        cur_state = env.getState()
                        for i in range(config.n_rollout_threads):
                            current_trajectory[i].append(cur_state[i])

                    
                    if et_i == 0:
                        total_eval = rewards
                    else:
                        total_eval = total_eval + rewards
                    obs = next_obs
                current_eval.append(total_eval)
                if config.store_traj:
                    for i in range(config.n_rollout_threads):
                        current_trajectories.append([current_entities[i], current_trajectory[i]])
            
            if config.store_traj:
                eval_trajectories.append(current_trajectories)
            evaluation_rewards.append(current_eval)

    if config.store_traj:
        with open(run_dir / 'static_trajectories.pkl', 'wb') as fp:
            pickle.dump(trajectories, fp)

        with open(run_dir / 'eval_static_trajectories.pkl', 'wb') as fp:
            pickle.dump(eval_trajectories, fp)


    if config.benchmark:
        with open(run_dir / 'info.pkl', 'wb') as fp:
            pickle.dump(agent_info, fp)

    with open(run_dir / 'rew.pkl', 'wb') as fp:
        pickle.dump(reward_info, fp)

    with open(run_dir / 'eval_rew.pkl', 'wb') as fp:
        pickle.dump(evaluation_rewards, fp)

    with open(run_dir / 'time.pkl', 'wb') as fp:
        pickle.dump(durations, fp)
    
    with open(run_dir / 'returns.pkl', 'wb') as fp:
        pickle.dump(total_returns, fp)
    
    if maddpg.gasil: 
        with open(run_dir / 'expert_average.pkl', 'wb') as fp:
            pickle.dump(expert_average_returns, fp)
        if config.store_traj:
            with open(run_dir / 'expert_trajectories.pkl', 'wb') as fp:
                pickle.dump(expert_trajectories, fp)

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--expert_buffer_length", default=int(25), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--n_eval_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=512, type=int,
                        help="Batch size for model training")
    parser.add_argument("--batch_size_disc",
                        default=256, type=int,
                        help="Batch size for model training")

    parser.add_argument("--num_disc_updates", default=4, type=int,
                        help="number of Discriminator mini batches")
    parser.add_argument("--num_AC_updates", default=16, type=int,
                        help="number of Critic, Policy mini batches")

    parser.add_argument("--num_disc_permutations", default=4, type=int,
                        help="number of discriminator permutations")

    parser.add_argument("--num_AC_permutations", default=4, type=int,
                        help="number of AC permutations")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--eval_interval", default=10, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--dlr", default=0.0003, type=float)
    parser.add_argument("--lambda_disc", default=0.5, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--rew_shape",
                        default=0, type=int,
                        choices=[0, 1, 2])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')
    parser.add_argument("--store_traj",
                        action='store_true')
    parser.add_argument("--sparse_reward",
                        action='store_true')
    parser.add_argument("--benchmark",
                        action='store_true')
    parser.add_argument("--stochastic",
                        action='store_true')
    parser.add_argument("--commonCritic",
                        action='store_true')
    parser.add_argument("--gasil",
                        action='store_true')
    parser.add_argument("--dynamic",
                        action='store_true')
    config = parser.parse_args()
    run(config)
