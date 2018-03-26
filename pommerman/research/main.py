import copy
import glob
import os
import time
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from model import CNNPolicy, MLPPolicy, PommeCNNPolicy, PommeResnetPolicy, PommeCNNPolicySmall
from storage import RolloutStorage
from visualize import visdom_plot

import numpy as np

# Add the library to the Python path so that we can import its modules
LIB_DIR = os.path.abspath(os.path.join("..", "games"))
if not LIB_DIR in sys.path:
    sys.path.append(LIB_DIR)

# import modules for Pommerman
from ..configs import create_game_config

args = get_args()

# num_updates = number of samples collected for one round of updates = number of updates in one round
# num_steps = horizon = number of steps in a rollout
# num_processes = number of parallel processes/workers that run the environment and collect data
# number of samples you use for a round of updates = number of collected samples to update each time = horizon * num_workers = num_steps_rollout * num_parallel_processes
num_updates = int(args.num_frames) // args.num_steps // args.num_processes
print("NUM UPDATES {} num frames {} num steps {} num processes {}".format(num_updates, args.num_frames, args.num_steps, args.num_processes), "\n")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom(server='http://216.165.70.24', port=8097)
        # viz = Visdom(port=args.port)
        win = None

    # Instantiate the environment
    config = create_game(args.config)

    # from ppo
    envs = [make_env(args, config, i) for i in range(args.num_processes)]
    envs = SubprocVecEnv(envs) if args.num_processes > 1 else DummyVecEnv(envs)
    if len(envs.observation_space[0].shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space[0].shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if len(envs.observation_space[0].shape) == 3:
        if args.model == 'convnet':
            actor_critic = [PommeCNNPolicySmall(obs_shape[0], envs.action_space, args) for i in range(args.nagents)]
        elif args.model == 'resnet':
            actor_critic = [PommeResnetPolicy(obs_shape[0], envs.action_space, args) for i in range(args.nagents)]
    else:
        actor_critic = [MLPPolicy(obs_shape[0], envs.action_space) for i in range(args.nagents)]
    # print("model ", args.model, "\n", actor_critic[0])

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        for i in range(args.nagents):
            actor_critic[i].cuda()

    optimizer = [optim.Adam(actor_critic[i].parameters(), args.lr, eps=args.eps) for i in range(args.nagents)]

    rollouts = [RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic[0].state_size) for _ in range(args.nagents)]
    current_obs = torch.zeros(args.nagents, args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space[0].shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :, :-shape_dim0] = current_obs[:, :, shape_dim0:]
        current_obs[:, :, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)
    rollouts.observations[0,:,:,:,:,:].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, args.nagents, 1])
    final_rewards = torch.zeros([args.num_processes, args.nagents, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []
            episode_reward = []
            cpu_actions_agents = [[-1 for k in range(args.nagents)] for i in range(args.num_processes)]
            for i in range(args.nagents):
                value, action, action_log_prob, states = actor_critic[i].act(Variable(rollouts.observations[step][i], volatile=True),
                                                                          Variable(rollouts.states[step][i], volatile=True),
                                                                          Variable(rollouts.masks[step][i], volatile=True))

                value_agents.append(value)
                action_agents.append(action)
                action_log_prob_agents.append(action_log_prob)
                states_agents.append(states)

                cpu_actions = action.data.squeeze(1).cpu().numpy()
                for k in range(args.num_processes):
                    cpu_actions_agents[k][i] = cpu_actions[k]

            obs, reward, done, info = envs.step(cpu_actions_agents)

            reward = torch.from_numpy(np.stack(reward)).float()

            episode_rewards += reward
            episode_rewards = episode_rewards.view(args.num_processes, args.nagents)

            # If done then clean the history of observations.
            # XXX: shouldn't we have done for each agent in the game so that we can only update
            # each agent in each episode as long as they are active - they don't all play the same
            # number of steps in one episode since some of them die earlier than others
            # they should also probably get the reward when they die
            # XXX: changed this so that it takes a mask for each agent in the game - corresponding to whether the game is over or not
            masks = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0] if done_ else [1.0, 1.0, 1.0, 1.0] for done_ in done])
            # masks = torch.FloatTensor([torch.zeros(args.nagents) if done_ else torch.ones(args.nagents) for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            if args.cuda:
                masks = masks.cuda()

            # transpose mask since it is num_processes x nagents and we need nagets x numprocesses for multiplying with current_obs
            reward_all = reward.transpose(0,1).unsqueeze(2)
            masks_all = masks.transpose(0,1).unsqueeze(2)    # masks: nagents x num_processes x 1
            # current_obs: nagents x num_processes x 13 x 23 x 23
            current_obs *= masks_all.unsqueeze(2).unsqueeze(2)  # masks: nagents x num_processes x 1 x 1 x 1
            update_current_obs(obs)

            states_all = torch.from_numpy(np.stack([x.data for x in states_agents])).float()
            action_all = torch.from_numpy(np.stack([x.data for x in action_agents])).float()
            action_log_prob_all = torch.from_numpy(np.stack([x.data for x in action_log_prob_agents])).float()
            value_all = torch.from_numpy(np.stack([x.data for x in value_agents])).float()

            rollouts.insert(step, current_obs, states_all, action_all, action_log_prob_all, value_all, reward_all, masks_all)

        next_value_agents = []
        for i in range(args.nagents):
            next_value = actor_critic[i](Variable(rollouts.observations[-1][i], volatile=True),
                                        Variable(rollouts.states[-1][i], volatile=True),
                                        Variable(rollouts.masks[-1][i], volatile=True))[0].data
            next_value_agents.append(next_value)

        rollouts.compute_returns(next_value_agents, args.use_gae, args.gamma, args.tau, args.nagents)

        # PPO
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for i in range(args.nagents):
            for e in range(args.ppo_epoch):
                data_generator = rollouts[i].feed_forward_generator(advantages, args.num_mini_batch, args)

                for sample in data_generator:
                    observations_batch, states_batch, actions_batch, \
                        return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, states = actor_critic[i].evaluate_actions(Variable(observations_batch),
                                                                                                      Variable(states_batch),
                                                                                                      Variable(masks_batch),
                                                                                                      Variable(actions_batch))

                    adv_targ = Variable(adv_targ)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                    value_loss = (Variable(return_batch) - values).pow(2).mean()

                    optimizer[i].zero_grad()
                    (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
                    nn.utils.clip_grad_norm(actor_critic[i].parameters(), args.max_grad_norm)
                    optimizer[i].step()

            rollouts[i].after_update()


        if j % args.save_interval == 0 and args.save_dir != "":
            try:
                os.makedirs(args.save_dir)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            for i in range(args.nagents):
                save_model = actor_critic[i]
                if args.cuda:
                    save_model = copy.deepcopy(actor_critic[i]).cpu()

                save_model = [save_model, hasattr(envs, 'ob_rms') and envs.ob_rms or None]
                torch.save(save_model, os.path.join(args.save_dir, args.env_name + "-agent_%d.pt" % i))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, 'ppo')
            except IOError:
                pass

if __name__ == "__main__":
    main()
