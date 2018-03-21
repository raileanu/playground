# pommerman_selfplay

### Notes:

- Can do without recurrent policy in favor of providing the agent with the last N frames (10?)


### TODO:
- [ ] Distribute across many GPUs.

- [ ] Fix bug in DummyVecEnv - for num_processes=1.

- [ ] Get a working pipeline for FFA (1 agent) against three SimpleAgents. We need to beat this.

- [ ] Get a working pipeline for Team Random doing self-play against itself. This should include having the code generate four training trajectories per run. 

- [ ] Try to understand the reward functions here. An initial simple one like 10 for winning agents that remain until the end, 5 for winning agent that dies earlier, and -7.5 for losing agents makes sense. However, that is probably insufficient as it penalizes strong play that barely loses, rewards weak player that dies early on, and is also too sparse. Consider adding dense rewards for:
  1. Picking up Items
  2. Killing enemies.
  3. Killing teammates (negative).
  4. Blowing up walls.

- [ ] Get a testing pipeline for the above where it plays against prior versions of itself and against SimpleAgent. This should be able to track its progress sufficiently.

References for Architecture
https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/OthelloNNet.py
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
https://github.com/rossumai/nochi/blob/master/michi/net.py
