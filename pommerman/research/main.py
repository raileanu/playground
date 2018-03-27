import copy
import glob
import os
import time
import sys

import gym
from pommerman import configs
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from model import CNNPolicy, MLPPolicy, PommeCNNPolicy, PommeResnetPolicy, PommeCNNPolicySmall
import ppo_agent
from visualize import visdom_plot


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
        viz = Visdom(port=args.port)

    # Instantiate the environment
    config = getattr(configs, args.config)()

    # We make this in order to get the shapes.
    dummy_env = make_env(args, config, -1, [config['agent'](game_type=config['game_type'])])()
    envs_shape = dummy_env.observation_space.shape[1:]
    obs_shape = (envs_shape[0], *envs_shape[1:])
    action_space = dummy_env.action_space
    if len(envs_shape) == 3:
        if args.model == 'convnet':
            actor_critic = lambda saved_model: PommeCNNPolicySmall(obs_shape[0], action_space, args)
        elif args.model == 'resnet':
            actor_critic = lambda saved_model: PommeResnetPolicy(obs_shape[0], action_space, args)
    else:
        actor_critic = lambda saved_model: MLPPolicy(obs_shape[0], action_space)

    # We need to get the agent = config.agent(agent_id, config.game_type) and then
    # pass that agent into the agent.PPOAgent
    training_agents = []
    saved_models = args.saved_models
    saved_models = saved_models.split(',') if saved_models else [None]*args.nagents
    assert(len(saved_models)) == args.nagents
    for saved_model in saved_models:
        # TODO: implement the model loading.
        model = actor_critic(saved_model)
        agent = config['agent'](game_type=config['game_type'])
        agent = ppo_agent.PPOAgent(agent, model)
        training_agents.append(agent)

    if args.how_train == 'simple':
        # Simple trains a single agent against three SimpleAgents.
        assert(args.nagents == 1), "Simple training should have a single agent."
        num_training_per_episode = 1
    elif args.how_train == 'homogenous':
        # Homogenous trains a single agent against itself (self-play).
        assert(args.nagents == 1), "Homogenous toraining should have a single agent."
        num_training_per_episode = 4
    elif args.how_train == 'heterogenous':
        assert(args.nagents > 1), "Heterogenous training should have more than one agent."
        print("Heterogenous training is not implemented yet.")
        return

    # NOTE: Does this actually work correctly? If the agents are shared, then will the threads operate independently?
    envs = [make_env(args, config, i, training_agents) for i in range(args.num_processes)]
    envs = SubprocVecEnv(envs) if args.num_processes > 1 else DummyVecEnv(envs)
    # TODO: Figure out how to render this for testing purposes. The following link may help:
    # https://github.com/MG2033/A2C/blob/master/envs/subproc_vec_env.py

    for agent in training_agents:
        agent.initialize(args, obs_shape, action_space, num_training_per_episode)

    current_obs = torch.zeros(args.num_processes, num_training_per_episode, *obs_shape)
    def update_current_obs(obs):
        current_obs = torch.from_numpy(obs).float()

    obs = envs.reset()
    update_current_obs(obs)
    if args.how_train == 'simple':
        training_agents[0].update_rollouts(obs=current_obs, timestep=0)
    elif args.how_train == 'homogenous':
        training_agents[0].update_rollouts(obs=current_obs, timestep=0)

    # These variables are used to compute average rewards for all processes.
    # TODO: Should this have 1 on the end?
    episode_rewards = torch.zeros([args.num_processes, num_training_per_episode, 1])
    final_rewards = torch.zeros([args.num_processes, num_training_per_episode, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        for agent in training_agents:
            agent.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            value_agents = []
            action_agents = []
            action_log_prob_agents = []
            states_agents = []
            episode_reward = []
            cpu_actions_agents = []

            if args.how_train == 'simple':
                value, action, action_log_prob, states = training_agents[0].act_pytorch(step, 0)
                value_agents.append(value)
                action_agents.append(action)
                action_log_prob_agents.append(action_log_prob)
                states_agents.append(states)
                cpu_actions = action.data.squeeze(1).cpu().numpy()
                cpu_actions_agents = cpu_actions
            elif args.how_train == 'homogenous':
                # TODO: Unite this properly with the "simple" scenario.
                cpu_actions_agents = [[None for _ in range(4)] for i in range(args.num_processes)]
                for i in range(4):
                    value, action, action_log_prob, states = training_agents[0].act_pytorch(step, i)
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
            episode_rewards = episode_rewards.view(args.num_processes, num_training_per_episode)

            # If done, clean the history of observations.
            masks = torch.FloatTensor([
                [0.0]*num_training_per_episode if done_ else [1.0]*num_training_per_episode
                for done_ in done
            ])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            if args.cuda:
                masks = masks.cuda()

            reward_all = reward.unsqueeze(2)
            masks_all = masks.unsqueeze(2)
            # masks: nagents x num_processes x 1 x 1 x 1
            current_obs *= masks_all.unsqueeze(2).unsqueeze(2)  
            update_current_obs(obs)

            states_all = torch.from_numpy(np.stack([x.data for x in states_agents])).float()
            action_all = torch.from_numpy(np.stack([x.data for x in action_agents])).float()
            action_log_prob_all = torch.from_numpy(np.stack([x.data for x in action_log_prob_agents])).float()
            value_all = torch.from_numpy(np.stack([x.data for x in value_agents])).float()

            if args.how_train == 'simple':
                training_agents[0].insert_rollouts(
                    step, current_obs, states_all, action_all, action_log_prob_all,
                    value_all, reward_all, masks_all)
            elif args.how_train == 'homogenous':
                # TODO
                pass

        # TODO: Fix below. 
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
