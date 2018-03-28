import torch
import torch.nn as nn


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

# A temporary solution from the master branch.
# https://github.com/pytorch/pytorch/blob/7752fe5d4e50052b3b0bbc9109e599f8157febc0/torch/nn/init.py#L312
# Remove after the next version of PyTorch gets release.
def orthogonal(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor


# initialize training stats
def init_stats(args):
    stats = {
             'num_updates'    : [],
             'num_steps'    : [],
             'mean_reward'  : [[] for i in range(args.nagents)],
             'median_reward'  : [[] for i in range(args.nagents)],
             'min_reward'  : [[] for i in range(args.nagents)],
             'max_reward'  : [[] for i in range(args.nagents)],
             'policy_loss'   : [[] for i in range(args.nagents)],
             'value_loss' : [[] for i in range(args.nagents)],
             'entropy_loss' : [[] for i in range(args.nagents)],
             }

    return stats

# save a dictionary with all the logs
def save_dict(fname, d):
    f = h5py.File(fname, "w")
    for k, v in d.items():
        f.create_dataset(k, data=v)
    f.close()


# log training stats
def log_stats(args, stats, num_updates, num_steps, mean_reward, median_reward, min_reward, max_reward, policy_loss, value_loss, entropy_loss):

    stats['num_updates'].append(num_updates)
    stats['num_steps'].append(num_steps)

    for i in range(args.nagents):
        stats['mean_reward'][i].append(mean_reward[i])
        stats['median_reward'][i].append(median_reward[i])
        stats['min_reward'][i].append(min_reward[i])
        stats['max_reward'][i].append(max_reward[i])

        stats['policy_loss'][i].append(policy_loss[i])
        stats['value_loss'][i].append(value_loss[i])
        stats['entropy_loss'][i].append(entropy_loss[i])

    return stats
