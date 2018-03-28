import os
import sys

import gym
from gym.spaces.box import Box

from baselines import bench

import numpy as np


# Add the library to the Python path so that we can import its modules
# LIB_DIR = os.path.abspath(os.path.join("../..", "pommerman"))
# if not LIB_DIR in sys.path:
#     sys.path.append(LIB_DIR)


# from agents import SimpleAgent
# from . import agents

# TODO: fix bug here
from .agents import SimpleAgent


def make_env(args, config, rank):
    def _thunk():
        env = config.env(**config["env_kwargs"])
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

    return _thunk


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
