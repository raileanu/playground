"""
PPO Agent using 
"""
from pommerman.agents import BaseAgent
from torch.autograd import Variable
import torch.optim as optim

from storage import RolloutStorage


class PPOAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""
    def __init__(self, agent, actor_critic):
        self._actor_critic = actor_critic
        self._agent = agent

    def cuda(self):
        self._actor_critic.cuda()
        self._rollout.cuda()

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions."""
        return None

    def act_pytorch(self, step, num_agent):
        return self._actor_critic.act(
            Variable(self._rollout.observations[step, :, num_agent], volatile=True),
            Variable(self._rollout.states[step, :, num_agent], volatile=True),
            Variable(self._rollout.masks[step, :, num_agent], volatile=True)
        )

    def initialize(self, args, obs_shape, action_space, num_training_per_episode):
        self._optimizer = optim.Adam(self._actor_critic.parameters(), args.lr, eps=args.eps)
        self._rollout = RolloutStorage(
            args.num_steps, args.num_processes, obs_shape, action_space,
            self._actor_critic.state_size, num_training_per_episode
        )

    def update_rollouts(self, obs, timestep):
        print("UPDATEROLL: ", obs.shape) # np,na,25*ns,13,13
        print(self._rollout.observations[timestep].shape) # na,np,25*ns,13,13
        self._rollout.observations[timestep, :, :, :, :, :].copy_(obs)

    def insert_rollouts(self, step, current_obs, states, action, action_log_prob,
                        value, reward, mask):
        self._rollout.insert(step, current_obs, states, action, action_log_prob, value, reward, mask)

    def copy(self, agent):
        # NOTE: Ugh.
        return PPOAgent(agent, None)

