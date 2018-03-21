# pommerman_selfplay


### TODO:
- [ ] do we need to distribute across many gpus?

- [ ] fix recurrent_generator - similar to feed_forward one

- [ ] make sure the recurrent architecture also WORKS

- [ ] do we want recurrent architecture?

- [ ] do we want to take multiple (past) frames as input?

- [ ] fix bug in DummyVecEnv - for num_processes=1

References for Architecture
https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/OthelloNNet.py
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
https://github.com/rossumai/nochi/blob/master/michi/net.py
