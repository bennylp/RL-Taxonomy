# RL Taxonomy

This is a loose taxonomy of RL algorithms. I'm by no means expert in this area, so please PR to correct things or suggest new stuff.

#### Table of Contents:<HR>

[Reinforcement Learning](#ReinforcementLearning)
  - [Value Gradient](#ValueGradient)
    - [SARSA](#SARSA)
    - [Q-learning](#Qlearning)
    - [DQN](#DQN)
    - [DRQN](#DRQN)
    - [DDQN](#DDQN)
    - [PER](#PER)
    - [Duelling-DQN](#DuellingDQN)
    - [QR-DQN](#QRDQN)
    - [RAINBOW](#RAINBOW)
    - [DQN+HER](#DQNHER)
  - [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic)
    - [REINFORCE](#REINFORCE)
    - [DPG](#DPG)
    - [DDPG](#DDPG)
    - [TRPO](#TRPO)
    - [GAE](#GAE)
    - [A3C](#A3C)
    - [DDPG+HER](#DDPGHER)
    - [MADDPG](#MADDPG)
    - [A2C](#A2C)
    - [ACER](#ACER)
    - [ACKTR](#ACKTR)
    - [PPO](#PPO)
    - [SVPG](#SVPG)
    - [D4PG](#D4PG)
    - [SAC](#SAC)
    - [TD3](#TD3)
    - [IMPALA](#IMPALA)

## Taxonomy
    
![RL Taxonomy](rl-taxonomy.gv.svg "RL Taxonomy")

## <a name="ReinforcementLearning"></a>Reinforcement Learning
Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward [from Wikipedia]

- Related to subsequent ideas:
  - [Value Gradient](#ValueGradient)
  - [Policy Gradient](#PolicyGradient)
- Useful links:
  - [A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)
  - [Reinforcement Learning: An Introduction - 2nd Edition (book) - Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book.html)
- Videos:
  - [All Reinforcement Courses on YouTube](https://www.youtube.com/playlist?list=PLvan4zSb2Rar-jNURaahH81uMyKHkGcHS)

### <a name="ValueGradient"></a>Value Gradient
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Value Gradient](#ValueGradient))

The algorithm is learning the value function of each state or state-action. The policy is implicit, usually by just selecting the best value


#### <a name="SARSA"></a>SARSA
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Value Gradient](#ValueGradient) --> [SARSA](#SARSA))

SARSA (State-Action-Reward-State-Action) is an on-policy TD control method

- Paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.2539&rep=rep1&type=pdf
- Authors: G. A. Rummery, M. Niranjan
- Year: 1994
- Flags:
  - On-Policy (ONP)
  - Discrete action space (DA)
- Related to prior idea:
  - [Value Gradient](#ValueGradient)

#### <a name="Qlearning"></a>Q-learning
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Value Gradient](#ValueGradient) --> [Q-learning](#Qlearning))

Q-learning an off-policy TD control method. Unlike SARSA, it doesn't follow the policy to find the next action but rather chooses most optimal action in a greedy fashion

- Paper: http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf
- Authors: Chris Watkins
- Year: 1989
- Flags:
  - Off-Policy (OFP)
  - Discrete action space (DA)
- Related to prior idea:
  - [Value Gradient](#ValueGradient)
- Related to subsequent idea:
  - [DQN](#DQN)
- Useful links:
  - [freecodecamp.org](https://www.freecodecamp.org/news/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe/)
  - [medium.com](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)

#### <a name="DQN"></a>DQN
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Value Gradient](#ValueGradient) --> [DQN](#DQN))

Deep Q Network. Q-Learning with using deep neural network as value estimator

- Paper: https://arxiv.org/abs/1312.5602
- Authors: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller
- Year: 2013
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Discrete action space (DA)
  - Replay Buffer (RB)
- Related to prior idea:
  - [Q-learning](#Qlearning)
- Related to subsequent ideas:
  - [DRQN](#DRQN)
  - [DDQN](#DDQN)
  - [PER](#PER)
  - [QR-DQN](#QRDQN)
  - [DQN+HER](#DQNHER)
  - [DDPG](#DDPG) (replay buffer)
  - [ACER](#ACER) (replay buffer, workers)
- Useful links:
  - [towardsdatascience.com](https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f)
  - [freecodecamp.org](https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/)

#### <a name="DRQN"></a>DRQN
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Value Gradient](#ValueGradient) --> [DRQN](#DRQN))

Deep Recurrent Q-Learning. Adding recurrency to a Deep Q-Network (DQN) by replacing the first post-convolutional fully-connected layer with a recurrent LSTM

- Paper: https://arxiv.org/abs/1507.06527
- Authors: Matthew Hausknecht, Peter Stone
- Year: 2015
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Discrete action space (DA)
  - Replay Buffer (RB)
  - Recurrent Neural Network (RNN)
- Related to prior idea:
  - [DQN](#DQN)

#### <a name="DDQN"></a>DDQN
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Value Gradient](#ValueGradient) --> [DDQN](#DDQN))

Double DQN adds another neural network, making separate network for policy and target. The target network is only updated after certain number of steps/episodes. This makes the learning more stable.

- Paper: https://arxiv.org/abs/1509.06461
- Authors: Hado van Hasselt, Arthur Guez, David Silver
- Year: 2015
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Discrete action space (DA)
- Related to prior idea:
  - [DQN](#DQN)
- Related to subsequent ideas:
  - [Duelling-DQN](#DuellingDQN)
  - [RAINBOW](#RAINBOW)
  - [TD3](#TD3) (double Q-learning)
- Useful links:
  - [towardsdatascience.com](https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f)

#### <a name="PER"></a>PER
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Value Gradient](#ValueGradient) --> [PER](#PER))

Prioritized Experience Replay (PER) improves data efficiency by replaying transitions from which there is more to learn more often

- Paper: https://arxiv.org/abs/1511.05952
- Authors: Tom Schaul, John Quan, Ioannis Antonoglou, David Silver
- Year: 2015
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Discrete action space (DA)
  - Replay Buffer (RB)
- Related to prior idea:
  - [DQN](#DQN)
- Related to subsequent idea:
  - [RAINBOW](#RAINBOW)

#### <a name="DuellingDQN"></a>Duelling-DQN
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Value Gradient](#ValueGradient) --> [Duelling-DQN](#DuellingDQN))

Duelling DQN represents two separate estimators: one for the state value function and one for the state-dependent action advantage function. The main benefit of this factoring is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm.

- Paper: https://arxiv.org/abs/1511.06581
- Authors: Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas
- Year: 2016
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Discrete action space (DA)
- Related to prior idea:
  - [DDQN](#DDQN)
- Related to subsequent idea:
  - [RAINBOW](#RAINBOW)

#### <a name="QRDQN"></a>QR-DQN
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Value Gradient](#ValueGradient) --> [QR-DQN](#QRDQN))

Distributional Reinforcement Learning with Quantile Regression (QR-DQN). In QR-DQN, distribution of values values are used for each state-action pair instead of a single mean value

- Paper: https://arxiv.org/abs/1710.10044
- Authors: Will Dabney, Mark Rowland, Marc G. Bellemare, Rémi Munos
- Year: 2017
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Discrete action space (DA)
  - Replay Buffer (RB)
- Related to prior idea:
  - [DQN](#DQN)
- Related to subsequent idea:
  - [RAINBOW](#RAINBOW)
- Useful links:
  - [github.com](https://github.com/senya-ashukha/quantile-regression-dqn-pytorch)

#### <a name="RAINBOW"></a>RAINBOW
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Value Gradient](#ValueGradient) --> [RAINBOW](#RAINBOW))

Examines six extensions to the DQN algorithm and empirically studies their combination

- Paper: https://arxiv.org/abs/1710.02298
- Authors: Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver
- Year: 2017
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Discrete action space (DA)
  - Replay Buffer (RB)
- Related to prior ideas:
  - [DDQN](#DDQN)
  - [PER](#PER)
  - [Duelling-DQN](#DuellingDQN)
  - [QR-DQN](#QRDQN)
  - [A3C](#A3C)

#### <a name="DQNHER"></a>DQN+HER
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Value Gradient](#ValueGradient) --> [DQN+HER](#DQNHER))

DQN with Hindsight Experience Replay (HER)

- Paper: https://arxiv.org/abs/1707.01495
- Authors: Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, Wojciech Zaremba
- Year: 2017
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Discrete action space (DA)
  - Replay Buffer (RB)
- Related to prior idea:
  - [DQN](#DQN)
- Useful links:
  - [becominghuman.ai](https://becominghuman.ai/learning-from-mistakes-with-hindsight-experience-replay-547fce2b3305)

### <a name="PolicyGradientActorCritic"></a>Policy Gradient/Actor-Critic
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic))

The algorithm works directly to optimize the policy, with or without value function. If the value function is learned in addition to the policy, we would get Actor-Critic algorithm. Most policy gradient algorithms are Actor-Critic. The *Critic* updates value function parameters *w* and depending on the algorithm it could be action-value ***Q(a|s;w)*** or state-value ***V(s;w)***. The *Actor* updates policy parameters θ, in the direction suggested by the critic, ***π(a|s;θ)***. [from [Lilian Weng' blog](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)]

- Useful links:
  - [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
  - [RL — Policy Gradient Explained](https://medium.com/@jonathan_hui/rl-policy-gradients-explained-9b13b688b146)
  - [Going Deeper Into Reinforcement Learning: Fundamentals of Policy Gradients](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/)
  - [An introduction to Policy Gradients with Cartpole and Doom](https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/)

#### <a name="REINFORCE"></a>REINFORCE
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [REINFORCE](#REINFORCE))

REINFORCE (Monte-Carlo policy gradient) is a pure policy gradient algorithm that works without a value function. The agent collects a trajectory of one episode using its current policy, and uses the returns to update the policy parameter

- Paper: https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
- Authors: Ronald J. Williams
- Year: 1992
- Flags:
  - Monte Carlo (MC)
  - On-Policy (ONP)
  - Continuous state space (CS)
  - Discrete action space (DA)
- Related to prior idea:
  - [Policy Gradient](#PolicyGradient)
- Useful links:
  - [toronto.edu](http://www.cs.toronto.edu/~tingwuwang/REINFORCE.pdf)
  - [freecodecamp.org](https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/)

#### <a name="DPG"></a>DPG
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [DPG](#DPG))

Deterministic Policy Gradient. Abstract: In this paper we consider deterministic policy gradient algorithms for reinforcement learning with continuous actions. The deterministic policy gradient has a particularly appealing form: it is the expected gradient of the action-value function. This simple form means that the deterministic policy gradient can be estimated much more efficiently than the usual stochastic policy gradient. To ensure adequate exploration, we introduce an off-policy actor-critic algorithm that learns a deterministic target policy from an exploratory behaviour policy. We demonstrate that deterministic policy gradient algorithms can significantly outperform their stochastic counterparts in high-dimensional action spaces.

- Paper: http://proceedings.mlr.press/v32/silver14.pdf
- Authors: David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller
- Year: 2014
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Continuous action space (CA)
  - Deterministic Policy (DP)
- Related to prior idea:
  - [Policy Gradient](#PolicyGradient)
- Related to subsequent idea:
  - [DDPG](#DDPG)

#### <a name="DDPG"></a>DDPG
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [DDPG](#DDPG))

Deep Deterministic Policy Gradient (DDPG).

- Paper: https://arxiv.org/abs/1509.02971
- Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra
- Year: 2015
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Continuous action space (CA)
  - Deterministic Policy (DP)
  - Replay Buffer (RB)
- Related to prior ideas:
  - [DPG](#DPG)
  - [DQN](#DQN) (replay buffer)
- Related to subsequent ideas:
  - [DDPG+HER](#DDPGHER)
  - [MADDPG](#MADDPG)
  - [D4PG](#D4PG)
  - [TD3](#TD3)
- Useful links:
  - [openai.com](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
  - [github.io](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)

#### <a name="TRPO"></a>TRPO
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [TRPO](#TRPO))

Trust Region Policy Optimization (TRPO) improves training stability by enforcing a KL divergence constraint to avoid parameter updates that change the policy too much at one step.

- Paper: https://arxiv.org/pdf/1502.05477
- Authors: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel
- Year: 2015
- Flags:
  - On-Policy (ONP)
  - Continuous state space (CS)
  - Continuous action space (CA)
  - Advantage (ADV)
- Related to prior idea:
  - [Policy Gradient](#PolicyGradient)
- Related to subsequent ideas:
  - [GAE](#GAE)
  - [ACER](#ACER) (TRPO technique)
  - [PPO](#PPO)
- Useful links:
  - [medium.com](https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9)
  - [medium.com](https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-part-2-f51e3b2e373a)

#### <a name="GAE"></a>GAE
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [GAE](#GAE))

Generalized Advantage Estimation

- Paper: https://arxiv.org/abs/1506.02438
- Authors: John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel
- Year: 2015
- Flags:
  - On-Policy (ONP)
  - Continuous state space (CS)
  - Continuous action space (CA)
- Related to prior ideas:
  - [Policy Gradient](#PolicyGradient)
  - [TRPO](#TRPO)
- Useful links:
  - [Generalized Advantage Estimator Explained](https://notanymike.github.io/GAE/)
  - [Notes on the Generalized Advantage Estimation Paper](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/)

#### <a name="A3C"></a>A3C
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [A3C](#A3C))

Asynchronous Advantage Actor-Critic (A3C) is a classic policy gradient method with the special focus on parallel training. In A3C, the critics learn the state-value function, ***V(s;w)***, while multiple actors are trained in parallel and get synced with global parameters from time to time. Hence, A3C is good for parallel training by default, i.e. on one machine with multi-core CPU. [from [Lilian Weng' blog](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)]

- Paper: https://arxiv.org/abs/1602.01783
- Authors: Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu
- Year: 2016
- Flags:
  - On-Policy (ONP)
  - Continuous state space (CS)
  - Continuous action space (CA)
  - Advantage (ADV)
  - Stochastic Policy (SP)
- Related to prior idea:
  - [Policy Gradient](#PolicyGradient)
- Related to subsequent ideas:
  - [RAINBOW](#RAINBOW)
  - [A2C](#A2C)
  - [ACER](#ACER)
- Useful links:
  - [Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
  - [An implementation of A3C](https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c)

#### <a name="DDPGHER"></a>DDPG+HER
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [DDPG+HER](#DDPGHER))

Hindsight Experience Replay (HER)

- Paper: https://arxiv.org/abs/1707.01495
- Authors: Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, Wojciech Zaremba
- Year: 2017
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Discrete action space (DA)
  - Deterministic Policy (DP)
  - Replay Buffer (RB)
- Related to prior idea:
  - [DDPG](#DDPG)
- Useful links:
  - [becominghuman.ai](https://becominghuman.ai/learning-from-mistakes-with-hindsight-experience-replay-547fce2b3305)

#### <a name="MADDPG"></a>MADDPG
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [MADDPG](#MADDPG))

Multi-agent DDPG (MADDPG) extends DDPG to an environment where multiple agents are coordinating to complete tasks with only local information. In the viewpoint of one agent, the environment is non-stationary as policies of other agents are quickly upgraded and remain unknown. MADDPG is an actor-critic model redesigned particularly for handling such a changing environment and interactions between agents (from [Lilian Weng's blog](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#maddpg))

- Paper: https://arxiv.org/abs/1706.02275
- Authors: Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, Igor Mordatch
- Year: 2017
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Continuous action space (CA)
  - Deterministic Policy (DP)
  - Replay Buffer (RB)
- Related to prior idea:
  - [DDPG](#DDPG)

#### <a name="A2C"></a>A2C
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [A2C](#A2C))

A2C is a synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C). It uses multiple workers to avoid the use of a replay buffer.

- Paper: https://openai.com/blog/baselines-acktr-a2c/
- Authors: OpenAI
- Year: 2017
- Flags:
  - On-Policy (ONP)
  - Continuous state space (CS)
  - Continuous action space (CA)
  - Advantage (ADV)
  - Stochastic Policy (SP)
- Related to prior idea:
  - [A3C](#A3C)
- Related to subsequent ideas:
- Useful links:
  - [openai.com](https://openai.com/blog/baselines-acktr-a2c/)
  - [freecodecamp.org](https://www.freecodecamp.org/news/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d/)
  - [readthedocs.io](https://stable-baselines.readthedocs.io/en/master/modules/a2c.html)

#### <a name="ACER"></a>ACER
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [ACER](#ACER))

Actor-Critic with Experience Replay (ACER) combines several ideas of previous algorithms: it uses multiple workers (as A2C), implements a replay buffer (as in DQN), uses Retrace for Q-value estimation, importance sampling and a trust region. ACER is A3C's off-policy counterpart. ACER proposes several designs to overcome the major obstacle to making A3C off policy, that is how to control the stability of the off-policy estimator. (source: [Lilian Weng's blog](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#acer))

- Paper: https://arxiv.org/abs/1611.01224
- Authors: Ziyu Wang, Victor Bapst, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, Nando de Freitas
- Year: 2017
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Continuous action space (CA)
  - Advantage (ADV)
  - Replay Buffer (RB)
- Related to prior ideas:
  - [A3C](#A3C)
  - [DQN](#DQN) (replay buffer, workers)
  - [TRPO](#TRPO) (TRPO technique)

#### <a name="ACKTR"></a>ACKTR
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [ACKTR](#ACKTR))

Actor Critic using Kronecker-Factored Trust Region (ACKTR) is applying trust region optimization to deep reinforcement learning using a recently proposed Kronecker-factored approximation to the curvature.

- Paper: https://arxiv.org/abs/1708.05144
- Authors: Yuhuai Wu, Elman Mansimov, Shun Liao, Roger Grosse, Jimmy Ba
- Year: 2017
- Flags:
  - On-Policy (ONP)
  - Continuous state space (CS)
  - Continuous action space (CA)
  - Advantage (ADV)
- Related to prior ideas:
  - [Policy Gradient](#PolicyGradient)

#### <a name="PPO"></a>PPO
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [PPO](#PPO))

Proximal Policy Optimization (PPO) is similar to [TRPO](#TRPO) but uses simpler mechanism while retaining similar performance.

- Paper: https://arxiv.org/abs/1707.06347
- Authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
- Year: 2017
- Flags:
  - On-Policy (ONP)
  - Continuous state space (CS)
  - Discrete action space (DA)
  - Continuous action space (CA)
  - Advantage (ADV)
- Related to prior idea:
  - [TRPO](#TRPO)
- Related to subsequent idea:
- Useful links:
  - [openai.com](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
  - [openai.com](https://openai.com/blog/openai-baselines-ppo/)
- Videos:
  - [youtube.com](https://www.youtube.com/watch?v=5P7I-xPq8u8)

#### <a name="SVPG"></a>SVPG
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [SVPG](#SVPG))

Stein Variational Policy Gradient (SVPG)

- Paper: https://arxiv.org/abs/1704.02399
- Authors: Yang Liu, Prajit Ramachandran, Qiang Liu, Jian Peng
- Year: 2017
- Flags:
  - On-Policy (ONP)
  - Continuous state space (CS)
  - Discrete action space (DA)
  - Continuous action space (CA)
- Related to prior ideas:
  - [Policy Gradient](#PolicyGradient)
- Useful links:
  - [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#svpg)

#### <a name="D4PG"></a>D4PG
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [D4PG](#D4PG))

Distributed Distributional Deep Deterministic Policy Gradient (D4PG) adopts the very successful distributional perspective on reinforcement learning and adapts it to the continuous control setting. It combines this within a distributed framework. It also combines this technique with a number of additional, simple improvements such as the use of N-step returns and prioritized experience replay [from the paper's abstract]

- Paper: https://arxiv.org/abs/1804.08617
- Authors: Gabriel Barth-Maron, Matthew W. Hoffman, David Budden, Will Dabney, Dan Horgan, Dhruva TB, Alistair Muldal, Nicolas Heess, Timothy Lillicrap
- Year: 2018
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Continuous action space (CA)
  - Deterministic Policy (DP)
  - Replay Buffer (RB)
- Related to prior idea:
  - [DDPG](#DDPG)

#### <a name="SAC"></a>SAC
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [SAC](#SAC))

Soft Actor Critic (SAC) is an algorithm that optimizes a stochastic policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches.

- Paper: https://arxiv.org/abs/1801.01290
- Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
- Year: 2018
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Continuous action space (CA)
  - Continuous action space (CA)
  - Stochastic Policy (SP)
- Related to prior ideas:
  - [Policy Gradient](#PolicyGradient)
- Useful links:
  - [Spinning Up SAC page](https://spinningup.openai.com/en/latest/algorithms/sac.html)
  - [Author\s code](https://github.com/haarnoja/sac)

#### <a name="TD3"></a>TD3
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [TD3](#TD3))

Twin Delayed DDPG (TD3). TD3 addresses function approximation error in DDPG by introducing twin Q-value approximation network and less frequent updates

- Paper: https://arxiv.org/abs/1802.09477
- Authors: Scott Fujimoto, Herke van Hoof, David Meger
- Year: 2018
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Discrete action space (DA)
  - Deterministic Policy (DP)
  - Replay Buffer (RB)
- Related to prior ideas:
  - [DDPG](#DDPG)
  - [DDQN](#DDQN) (double Q-learning)
- Useful links:
  - [openai.com](https://spinningup.openai.com/en/latest/algorithms/td3.html)

#### <a name="IMPALA"></a>IMPALA
(Path: [Reinforcement Learning](#ReinforcementLearning) --> [Policy Gradient/Actor-Critic](#PolicyGradientActorCritic) --> [IMPALA](#IMPALA))

Importance Weighted Actor-Learner Architecture (IMPALA)

- Paper: https://arxiv.org/abs/1802.01561
- Authors: Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu
- Year: 2018
- Flags:
  - Off-Policy (OFP)
  - Continuous state space (CS)
  - Continuous action space (CA)
- Related to prior ideas:
  - [Policy Gradient](#PolicyGradient)
- Useful links:
  - [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)

<HR>
(This document is autogenerated)
