import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size, num_training_per_episode):
        self.observations = torch.zeros(num_steps + 1, num_processes, num_training_per_episode, *obs_shape)
        self.states = torch.zeros(num_steps + 1, num_processes, num_training_per_episode, state_size)
        self.rewards = torch.zeros(num_steps, num_processes, num_training_per_episode, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, num_training_per_episode, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, num_training_per_episode, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, num_training_per_episode, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, num_training_per_episode, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, num_training_per_episode, 1)

    def cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, current_obs, state, action, action_log_prob, value_pred, reward, mask):
        self.observations[step + 1].copy_(current_obs)
        self.states[step + 1].copy_(state)
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                                     gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch, args):
        num_steps = self.rewards.size(0)
        num_processes = self.rewards.size(2)

        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)

        for indices in sampler:
            indices = torch.LongTensor(indices)

            if advantages.is_cuda:
                indices = indices.cuda()

            # TODO: fix this bug
            observations_batch = self.observations[:-1].select(1).contiguous().view((args.num_steps*args.num_processes), *self.observations.size()[3:])[indices]
            states_batch = self.states[:-1].select(1).contiguous().view((args.num_steps*args.num_processes), 1)[indices]
            actions_batch = self.actions.select(1).contiguous().view((args.num_steps*args.num_processes), 1)[indices]
            return_batch = self.returns[:-1].select(1).contiguous().view((args.num_steps*args.num_processes), 1)[indices]
            masks_batch = self.masks[:-1].select(1).contiguous().view((args.num_steps*args.num_processes), 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.select(1).contiguous().view((args.num_steps*args.num_processes), 1)[indices]
            adv_targ = advantages.select(1).contiguous().view(-1, 1)[indices]

            yield observations_batch, states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ
