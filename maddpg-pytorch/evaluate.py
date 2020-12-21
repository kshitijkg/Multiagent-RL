import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
import pickle

def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    #model_path = config.path
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)
    
    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, config.benchmark, discrete_action=maddpg.discrete_action)
    print(type(env))
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    if config.save_gifs:
        frames = []
    agent_info = [[[]]]
    reward_info = []
    trajectories = []
    
    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        if config.save_gifs:
            frames.append(env.render('rgb_array')[0])
            env.render('human')
        episode_rewards = np.zeros((config.episode_length, maddpg.nagents))
        current_trajectory = []
        current_entities = []
        if config.store_traj:
            cur_state_ent = env.getStateEntities()
            current_entities.append(cur_state_ent)
            cur_state = env.getState()
            current_trajectory.append(cur_state)
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            if config.store_traj:
                cur_state = env.getState()
                current_trajectory.append(cur_state)

            if config.benchmark:
                for i, info in enumerate(infos):
                    agent_info[-1][i].append(infos['n'])
            if config.sparse_reward:
                if t_i == 0:
                    total = np.array(rewards)
                if t_i!=config.episode_length-1:
                    total = total + np.array(rewards)
                    rewards = list(np.zeros(len(rewards)))
                else:
                    rewards = list(total)
            episode_rewards[t_i] = rewards
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if config.save_gifs:
                if elapsed < ifi:
                    time.sleep(ifi - elapsed)
                env.render('human')
        agent_info.append([[]])
        mean_rewards = np.mean(episode_rewards, axis=0)
        reward_info.append(mean_rewards)
        if config.store_traj:
            trajectories.append([current_entities, current_trajectory])
    if config.save_gifs:
        gif_num = 0
        while (gif_path / ('%i.gif' % gif_num)).exists():
            gif_num += 1
        imageio.mimsave(str(gif_path / ('%i.gif' % gif_num)),
                        frames, duration=ifi)
    run_dir = model_path.parent 
    if config.benchmark:
        with open(run_dir / 'eval_info.pkl', 'wb') as fp:
            pickle.dump(agent_info, fp)
    with open(run_dir / 'eval_rew.pkl', 'wb') as fp:
        pickle.dump(reward_info, fp)
    if config.store_traj:
        with open(run_dir / 'static_trajectories_eval.pkl', 'wb') as fp:
            pickle.dump(trajectories, fp)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("path", help="path of experiment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--sparse_reward",
                        action='store_true')
    parser.add_argument("--benchmark",
                        action='store_true')
    parser.add_argument("--store_traj",
                        action='store_true')
    config = parser.parse_args()

    run(config)