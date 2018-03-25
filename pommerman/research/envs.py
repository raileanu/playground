import os

import gym
from gym.spaces.box import Box

from baselines import bench
<<<<<<< HEAD
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

try:
    import pybullet_envs
    import roboschool
except ImportError:
    pass
=======
>>>>>>> upstream/master

import numpy as np


# for Pommerman
<<<<<<< HEAD
from a.pommerman.envs.v0 import Pomme
from a.agents import RandomAgent
from a.pommerman.agents import SimpleAgent
from gym import spaces

# from tensorforce.agents import PPOAgent
# from tensorforce.execution import Runner
# from tensorforce.contrib.openai_gym import OpenAIGym

# pommerman
def make_env(args, config, rank):
    def _thunk():
        env = Pomme(**config["env_kwargs"])
=======
from a.pommerman.agents import SimpleAgent

def make_env(args, config, rank):
    def _thunk():
        env = config.env(**config["env_kwargs"])
>>>>>>> upstream/master
        env.seed(args.seed + rank)

        agents = {}
        for agent_id in range(args.nagents):
            agents[agent_id] = SimpleAgent(config["agent"](agent_id, config["game_type"]))
        env.set_agents(list(agents.values()))

        # XXX: should we use the monitor or not? bug when using it - in sum(self.rewards) line 64
        # if args.log_dir is not None:
        #     env = bench.Monitor(env, os.path.join(args.log_dir, str(rank)))

        env = WrapPomme(env)

        return env

<<<<<<< HEAD
    # _thunk()
    # uncomment below when done debugging
    # _thunk() # for debugging and printing in the _thunk fct: to make sure it creates the environment and there are no errors in the _thunk function
    # _thunk()
    return _thunk

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None, obs_shape=(23,13,13)):
        super(WrapPyTorch, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [obs_shape[0], obs_shape[1], obs_shape[2]]
        )

    # XXX: what is the role of this?
    def _observation(self, observation):
        # XXX: need the agent_id to be able to do this
        observation_feat = featurize3D(observation[0])  # XXX: only first agent - need to extend to 4 agents
        return observation_feat#.transpose(2, 0, 1)
=======
    return _thunk

>>>>>>> upstream/master

# similar to PPO - no need to reset in here or anything
# TODO: make obs_shape and others arguments
class WrapPomme(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPomme, self).__init__(env)

        obs_shape = (23,13,13)           # XXX: make this an argument
        self.observation_space = [Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [obs_shape[0], obs_shape[1], obs_shape[2]]) for i in range(4)]

    # XXX: featurizes the observation space from dictionary to feature maps whenever you get to observe the environment
    def _observation(self, observation):
        # XXX: need the agent_id to be able to do this
        observation_feat = [featurize3D(observation[i]) for i in range(4)]  # XXX: only first agent - need to extend to 4 agents
        return observation_feat#.transpose(2, 0, 1)


# added for Pommerman
def make_np_float(feature):
    return np.array(feature).astype(np.float32)

# create 3D feature maps for Pommerman
def featurize3D(obs, has_sep_board_feat=True, has_teammate_feat=True, has_enemies_feat=True):
    map_size = len(obs["board"])

    # XXX: if we want different feature maps for each item in the board such as:
    # rigid wall, wooden wall, bomb, flames, fog, extra bomb, \
    # extra bomb power up, increate range power up, can kick, skill puwer up, \
    # agent0, agent1, agent2, agent3
    # in the above order - total 14 features maps of 13x13 dims - so (14,13,13) array of 1/0
    if has_sep_board_feat:
        board = np.zeros((14, map_size, map_size))
        for i in range(14):
            mask = obs["board"] == i
            board[i][obs["board"] == i] = 1

    # single feature map for the board - ints
    else:
        board = obs["board"].astype(np.float32)
        board = board.reshape(1, map_size, map_size)

    # feature map with ints where bombs are corresponding to their blast range
    bombs = obs["bombs"].astype(np.float32)
    bombs = bombs.reshape(1, map_size, map_size)

    # position of self agent - 1 at corresp location
    position = np.zeros((map_size, map_size)).astype(np.float32)
    position[obs["position"][0], obs["position"][1]] = 1
    position = position.reshape(1, map_size, map_size)

    # ammo of self agent - constant feature map of corresp integer
    ammo = np.ones((map_size, map_size)).astype(np.float32)*obs["ammo"]
    ammo = ammo.reshape(1, map_size, map_size)

    # blast strength of self agent - constants feature map
    blast_strength = np.ones((map_size, map_size)).astype(np.float32)*obs["blast_strength"]
    blast_strength = blast_strength.reshape(1, map_size, map_size)

    # whether the agent can kick - constant feature map of 1 or 0
    if obs["can_kick"]:
        can_kick = np.ones((map_size, map_size)).astype(np.float32)
    else:
        can_kick = np.zeros((map_size, map_size)).astype(np.float32)
    can_kick = can_kick.reshape(1, map_size, map_size)

    # if we want to include a feature map corresponding to its teammate (definitely in team_v0 etc.)
    # XXX: may not want this in ffa_v0 since no_one is your team?
    if has_teammate_feat:
        if obs["teammate"] is not None:
            teammate = np.ones((map_size, map_size)).astype(np.float32)*obs["teammate"].value
        else:
            teammate = np.ones((map_size, map_size)).astype(np.float32)*(-1)
        teammate = teammate.reshape(1, map_size, map_size)

    # XXX: if we want to include a feature map for its enemies - similar to above
    if has_enemies_feat:
        enemies = np.zeros((3, map_size, map_size))
        for i in range(len(obs["enemies"])):
            enemies[i] = np.ones((map_size, map_size)).astype(np.float32)*obs["enemies"][i].value
        if obs["teammate"] is not None:
            enemies[2] = np.ones((map_size, map_size)).astype(np.float32)*(-1)


    if has_teammate_feat and has_enemies_feat:
        feature_maps = np.concatenate((board, bombs, position, ammo, blast_strength, can_kick, teammate, enemies))
    elif has_teammate_feat:
        feature_maps = np.concatenate((board, bombs, position, ammo, blast_strength, can_kick, teammate))
    elif has_enemies_feat:
        feature_maps = np.concatenate((board, bombs, position, ammo, blast_strength, can_kick, enemies))
    else:
        feature_maps = np.concatenate((board, bombs, position, ammo, blast_strength, can_kick))

    return feature_maps
<<<<<<< HEAD


def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bombs = obs["bombs"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((board, bombs, position, ammo, blast_strength, can_kick, teammate, enemies))

# changed OpenAIGym to gym.ObservationWrapper
class WrappedEnv(gym.ObservationWrapper):
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def execute(self, actions):
        if self.visualize:
            self.gym.render()

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = featurize(obs[3])
        return agent_obs


# ppo
# def make_env(env_id, seed, rank, log_dir):
#     def _thunk():
#         env = gym.make(env_id)
#         is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
#         if is_atari:
#             env = make_atari(env_id)
#         env.seed(seed + rank)
#         if log_dir is not None:
#             env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
#         if is_atari:
#             env = wrap_deepmind(env)
#         # If the input has shape (W,H,3), wrap for PyTorch convolutions
#         obs_shape = env.observation_space.shape
#         if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
#             env = WrapPyTorch(env)
#         print("env ppo ", env, type(env))
#         return env
#
#     return _thunk
#
#
# class WrapPyTorch(gym.ObservationWrapper):
#     def __init__(self, env=None):
#         super(WrapPyTorch, self).__init__(env)
#         obs_shape = self.observation_space.shape
#         self.observation_space = Box(
#             self.observation_space.low[0,0,0],
#             self.observation_space.high[0,0,0],
#             [obs_shape[2], obs_shape[1], obs_shape[0]]
#         )
#
#     def _observation(self, observation):
#         return observation.transpose(2, 0, 1)
=======
>>>>>>> upstream/master
