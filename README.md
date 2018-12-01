# rubik_RL

A script that creates a self-learning Rubik's cube solver. The approach uses reinforcement learning and a fully connected neural network, that learns utilizing the policy gradient algorithm. The project was originally inspired by this blog-post from Andrej Karpathy: http://karpathy.github.io/2016/05/31/rl/.

# Requirements

Script is done in Python 3. The approach uses only basic packages (numpy), and all feed forward and backwards propogation is programmed from scratch.

# Details

One can choose to use either the Pocket cube (2x2) or Rubik's cube (3x3). Further one can construct different networks by changing the initialization parameters. One can set the number of hidden layers to be between one and three, and their respective sizes. One can choose how often to print statistics on the learning process with the variable print_every_batch. One can choose to save networks and thus being able to continue running from a midway point later.

# Results

The empirically attained, best neural network was constructed using two hidden layers, of roughly 250 nodes each.

This network required around 300 batches (=30.000 episodes) to learn to solve the Pocket cube. At that point the machine solves almost 100% of the cubes presented to it. The following figure presents the dynamics for the Pocket cube.

<img src="/solved_dur_score.png" width="700">

The time required to learn to solve the Rubik's cube is much larger. After 35.000 batches, the solver temporarily solves the cube around 60% of the times. The following figure presents the dynamics for the Rubik's cube.

<img src="/solved_dur_score.png" width="700">
