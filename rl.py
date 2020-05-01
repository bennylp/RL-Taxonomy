# -*- coding: utf-8 -*-
from collections import OrderedDict
import copy
from enum import Enum
import re
import sys

WEAK_LINK = 'dotted'


class Flag(Enum):
    """
    These are various flags that can be attributed to an algorithm
    """
    
    #MF = "Model-Free" # is the default for all algorithms
    MB = "Model-Based"
    MC = "Monte Carlo"
    #TD = "Temporal Difference" # is the default for all algorithms
    # Policy:
    ONP = "On-Policy"
    OFP = "Off-Policy"
    # Action space:
    DA = "Discrete action space"
    CA = "Continuous action space"
    # State space:
    DS = "Discrete state space"
    CS = "Continuous state space"
    # Policy space
    SP = "Stochastic Policy"
    DP = "Deterministic Policy"
    # Operator:
    QV = "Q-Value"
    ADV = "Advantage"
    # Miscellaneous:
    RB = "Replay Buffer"
      

def link_name(url):
    """
    Get suitable text for a given an URL (hint: it's the domain name)
    """
    o = re.search('//([^/]+)/', url)
    if not o:
        return 'link'
    parts = o.group(1).split('.')
    name = parts[-2] + '.' + parts[-1]
    return f'{name}'



class Node:
    """
    A Node represents an algorithm. The relevant properties can be initialized from
    the constructor.
    """
    
    def __init__(self, title, description, flags=[], authors=None, year=None, url=None,
                 videos=[], links=[], rank=0, **dot_kwargs):
        self.title = title
        self.description = description
        self._rank = rank
        self.flags = flags
        self.authors = authors
        self.year = year
        self.url = url
        self.videos = videos
        self.links = links
        self.children = []
        self.dot_kwargs = dot_kwargs
    
    @property
    def name(self):
        return re.sub('[^0-9a-zA-Z]+', '', self.title)
    
    @property
    def rank(self):
        return self.year or self._rank
    
    def add_to_graph(self, graph):
        fields = []
        if self.flags:
            fields += [f.name for f in self.flags]
        label = self.title
        if fields:
            label += f'|{{{"|".join(fields)}}}'
        graph.node(self.title, label=f'{{{label}}}', shape='record', style='rounded', 
                   **self.dot_kwargs)

    def to_md(self, with_children=True, level=1, parents=[]):
        md = ('#' * min(level,4)) + f' <a name="{self.name}"></a>{self.title}\n'
        parents.append(self)
        #md += f'(Path: '
        #md += ' --> '.join([f'[{p.title}](#{p.name})' for p in parents]) + ')\n\n'
        if self.description:
            md += f'{self.description}\n\n'
        if self.authors:
            md += f'- Authors: {self.authors}\n'
        if self.year:
            md += f'- Year: {self.year}\n'
        if self.url:
            md += f'- Paper: {self.url}\n'
        if self.links:
            md += '- Useful links: ' + ', '.join([f'[{link_name(l)}]({l})' for l in self.links]) + '\n'
        if self.videos:
            md += '- Videos: ' + ', '.join([f'[{link_name(l)}]({l})' for l in self.videos]) + '\n'
        if self.flags:
            md += '- Flags:\n'
            for f in self.flags:
                md += f'  - {f.value} ({f.name})\n'
        md += '\n'
        
        if with_children:
            for n in self.children:
                md += n.to_md(with_children=True, level=level+1, parents=parents)
            
        return md
    

class Edge:
    """
    An Edge is a connection between two Nodes
    """
    
    def __init__(self, n1, n2, **dot_kwargs):
        self.n1 = n1
        self.n2 = n2
        if self.n2 not in self.n1.children and dot_kwargs.get('style', '') != WEAK_LINK:
            self.n1.children.append(self.n2)
        self.dot_kwargs = dot_kwargs

    def add_to_graph(self, graph):
        graph.edge(self.n1.title, self.n2.title, **self.dot_kwargs)


edges = []

#
# Top-level
#    
rl = Node('Reinforcement Learning', 
          'Reinforcement learning (RL) is an area of machine learning concerned with how software '
          'agents ought to take actions in an environment in order to maximize the notion of cumulative '
          'reward [from Wikipedia]',
          rank=1)

value_based = Node('Value Based', 
                   'The algorithm contains value function to estimate the value of each state or state-action. '
                   'The policy is implicit, usually by just selecting the best value', 
                   flags=[],
                   rank=5)

policy_based = Node('Policy Based', 
                    'The algorithm stores the policy and works directly to optimize the policy without '
                    'estimating the state or state-action values', 
                    flags=[],
                    links=[
                        'https://medium.com/@jonathan_hui/rl-policy-gradients-explained-9b13b688b146',
                        'https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/'
                        ],
                    rank=5)

actor_critic = Node('Actor Critic', 
                    'Combination of policy-based and value-based. The agent works by directly optimizing '
                    'the policy, but by using value function to measure how good the policy is', 
                    flags=[],
                    rank=5)

edges += [Edge(rl, value_based),
          Edge(rl, actor_critic),
          Edge(rl, policy_based)
         ] 

#
# VALUE BASED
#
sarsa = Node('SARSA',
           'SARSA',
           flags=[Flag.ONP, Flag.DA],
           authors='G. A. Rummery, M. Niranjan',
           year=1994, 
           url='http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.2539&rep=rep1&type=pdf')

qlearning = Node('Q-learning',
           'Q-learning',
           flags=[Flag.OFP, Flag.DA, Flag.QV],
           authors='Chris Watkins',
           year=1989, 
           url='http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf',
           links=['https://www.freecodecamp.org/news/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe/',
                  'https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0']
           )

dqn = Node('DQN',
           'Deep Q Network. Q-Learning with using deep neural network as value estimator',
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.QV, Flag.RB],
           authors='Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller',
           year=2013, 
           url='https://arxiv.org/abs/1312.5602',
           links=['https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f',
                  'https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/'])

ddqn = Node('DDQN',
            'Double DQN adds another neural network, making separate network for policy and target. The target network is only updated after certain number of steps/episodes. This makes the learning more stable.',
            flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.QV],
            authors='Hado van Hasselt, Arthur Guez, David Silver',
            year=2015,
            url='https://arxiv.org/abs/1509.06461',
            links=['https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f'])

duel_dqn = Node('Duelling-DQN',
                'Duelling DQN',
                flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.QV],
                authors='Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas',
                year=2016, 
                url='https://arxiv.org/abs/1511.06581')

qr_dqn = Node('QR-DQN',
           'Distributional Reinforcement Learning with Quantile Regression (QR-DQN). In QR-DQN, '
           'distribution of values values are used for each state-action pair instead of a single mean value',
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.QV, Flag.RB],
           authors='Will Dabney, Mark Rowland, Marc G. Bellemare, Rémi Munos',
           year=2017, 
           url='https://arxiv.org/abs/1710.10044',
           links=['https://github.com/senya-ashukha/quantile-regression-dqn-pytorch'])

dqn_her = Node('DQN+HER',
           'Hindsight Experience Replay (HER)',
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.QV, Flag.RB],
           authors='Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, Wojciech Zaremba',
           year=2017, 
           url='https://arxiv.org/abs/1707.01495',
           links=['https://becominghuman.ai/learning-from-mistakes-with-hindsight-experience-replay-547fce2b3305'])

edges += [Edge(value_based, sarsa),
          Edge(value_based, qlearning),
          Edge(qlearning, dqn),
          Edge(dqn, ddqn),
          Edge(ddqn, duel_dqn),
          Edge(dqn, qr_dqn),
          Edge(dqn, dqn_her, style=WEAK_LINK),
         ]


#
# POLICY BASED
#
reinforce = Node('REINFORCE',
           'REINFORCE',
           flags=[Flag.MC, Flag.ONP, Flag.CS, Flag.DA],
           authors='Ronald J. Williams',
           year=1992, 
           url='https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf',
           links=['https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/']
           )
"""
vpg = Node('VPG',
           'Vanilla Policy Gradient',
           flags=[Flag.MC, Flag.ONP, Flag.CS, Flag.DA, Flag.CA],
           authors='Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour',
           year=2000, 
           url='https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf',
           links=['https://spinningup.openai.com/en/latest/algorithms/vpg.html']
           )
"""
edges += [Edge(policy_based, reinforce)]

#
# ACTOR-CRITIC
#
dpg = Node('DPG',
           'Deterministic Policy Gradient. Abstract: In this paper we consider deterministic policy gradient '
           'algorithms for reinforcement learning with continuous actions. The deterministic policy gradient '
           'has a particularly appealing form: it is the expected gradient of the action-value function. This '
           'simple form means that the deterministic policy gradient can be estimated much more efficiently than '
           'the usual stochastic policy gradient. To ensure adequate exploration, we introduce an off-policy '
           'actor-critic algorithm that learns a deterministic target policy from an exploratory behaviour policy. '
           'We demonstrate that deterministic policy gradient algorithms can significantly outperform their '
           'stochastic counterparts in high-dimensional action spaces.',
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.DP],
           authors='David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller',
           year=2014, 
           url='http://proceedings.mlr.press/v32/silver14.pdf',
           links=[]
           )

ddpg = Node('DDPG',
           'Deep Deterministic Policy Gradient.',
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.DP, Flag.RB],
           authors='Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra',
           year=2015, 
           url='https://arxiv.org/abs/1509.02971',
           links=['https://spinningup.openai.com/en/latest/algorithms/ddpg.html',
                  'https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html']
           )

ddpg_her = Node('DDPG+HER',
           'Hindsight Experience Replay (HER)',
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.DP, Flag.QV, Flag.RB],
           authors='Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, Wojciech Zaremba',
           year=2017, 
           url='https://arxiv.org/abs/1707.01495',
           links=['https://becominghuman.ai/learning-from-mistakes-with-hindsight-experience-replay-547fce2b3305'])

td3 = Node('TD3',
           'Twin Delayed DDPG (TD3). TD3 addresses function approximation error in DDPG by introducing twin '
           'Q-value approximation network and less frequent updates',
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.DP, Flag.QV, Flag.RB],
           authors='Scott Fujimoto, Herke van Hoof, David Meger',
           year=2018, 
           url='https://arxiv.org/abs/1802.09477',
           links=['https://spinningup.openai.com/en/latest/algorithms/td3.html'])

trpo = Node('TRPO',
           'Trust Region Policy Optimization',
           flags=[Flag.ONP, Flag.CS, Flag.CA, Flag.ADV],
           authors='John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel',
           year=2015, 
           url='https://arxiv.org/pdf/1502.05477',
           links=['https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9',
                  'https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-part-2-f51e3b2e373a']
           )

a3c = Node('A3C',
           'Asynchronous Advantage Actor-Critic (A3C)',
           flags=[Flag.ONP, Flag.CS, Flag.CA, Flag.ADV],
           authors='Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu',
           year=2016, 
           url='https://arxiv.org/abs/1602.01783',
           links=['https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2']
           )

a2c = Node('A2C',
           'A2C is a synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C). '
           'It uses multiple workers to avoid the use of a replay buffer.',
           flags=[Flag.ONP, Flag.CS, Flag.CA, Flag.ADV],
           authors='OpenAI',
           year=2017, 
           url='https://openai.com/blog/baselines-acktr-a2c/',
           links=[
               'https://openai.com/blog/baselines-acktr-a2c/',
               'https://www.freecodecamp.org/news/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d/'
               ]
           )

acer = Node('ACER',
           'Sample Efficient Actor-Critic with Experience Replay (ACER) combines several ideas of previous algorithms: it uses multiple workers (as A2C), implements a replay buffer (as in DQN), uses Retrace for Q-value estimation, importance sampling and a trust region.',
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.ADV, Flag.RB],
           authors='Ziyu Wang, Victor Bapst, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, Nando de Freitas',
           year=2017, 
           url='https://arxiv.org/abs/1611.01224',
           links=[
               ]
           )

acktr = Node('ACKTR',
           'Actor Critic using Kronecker-Factored Trust Region (ACKTR) is applying trust region optimization to deep reinforcement learning using a recently proposed Kronecker-factored approximation to the curvature.',
           flags=[Flag.ONP, Flag.CS, Flag.CA, Flag.ADV],
           authors='Yuhuai Wu, Elman Mansimov, Shun Liao, Roger Grosse, Jimmy Ba',
           year=2017, 
           url='https://arxiv.org/abs/1708.05144',
           links=[
               ]
           )

ppo = Node('PPO',
           'Proximal Policy Optimization. We\'re releasing a new class of reinforcement learning algorithms, '
           'Proximal Policy Optimization (PPO), which perform comparably or better than state-of-the-art '
           'approaches while being much simpler to implement and tune. PPO has become the default reinforcement '
           'learning algorithm at OpenAI because of its ease of use and good performance (OpenAI)',
           flags=[Flag.ONP, Flag.CS, Flag.DA, Flag.CA, Flag.ADV],
           authors='John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov',
           year=2017, 
           url='https://arxiv.org/abs/1707.06347',
           links=['https://spinningup.openai.com/en/latest/algorithms/ppo.html',
                  'https://openai.com/blog/openai-baselines-ppo/'],
           videos=['https://www.youtube.com/watch?v=5P7I-xPq8u8']
           )

sac = Node('SAC',
           'Soft Actor Critic (SAC) is an algorithm that optimizes a stochastic policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches.',
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.CA, Flag.QV, Flag.SP],
           authors='Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine',
           year=2018, 
           url='https://arxiv.org/abs/1801.01290',
           links=['https://spinningup.openai.com/en/latest/algorithms/sac.html'])

edges += [Edge(actor_critic, dpg),
          Edge(dpg, ddpg),
          Edge(dqn, ddpg, style=WEAK_LINK, label='replay buffer'),
          Edge(ddpg, ddpg_her, style=WEAK_LINK),
          Edge(ddpg, td3),
          Edge(actor_critic, trpo),
          Edge(actor_critic, a3c),
          Edge(a3c, a2c),
          Edge(a3c, acer),
          Edge(dqn, acer, style=WEAK_LINK, label='replay buffer'),
          Edge(actor_critic, acktr),
          #Edge(a2c, acktr, style='invisible', arrowhead='none'),
          Edge(actor_critic, ppo),
          Edge(trpo, ppo, style=WEAK_LINK),
          Edge(actor_critic, sac),
          #Edge(td3, sac, style='invisible', arrowhead='none'),
          ]


def collect_nodes():
    all_nodes = [v for v in globals().values() if isinstance(v, Node)]

    has_err = False
    names = []
    graph_nodes = OrderedDict()
    for e in edges:
        if e.n2 not in e.n1.children and e.n2 not in graph_nodes:
            e.n1.children.append(e.n2)
        graph_nodes[e.n1] = 1
        graph_nodes[e.n2] = 1
        if e.n2.name in names:
            sys.stderr.write(f'Warning: node "{e.n2.title}" was seen before\n')
        names += [e.n1.name, e.n2.name]
    
    graph_nodes = graph_nodes.keys()
    
    # Check all nodes are included in graph
    for node in all_nodes:
        if node not in graph_nodes:
            sys.stderr.write(f'Error: node "{node.title}" is not in graph\n')
            has_err = True
    if has_err:
        sys.exit(1)
        
    return all_nodes
    

def generate_dot(output, format='svg'):
    from graphviz import Digraph, Source
    
    nodes = collect_nodes()
    nodes = sorted(nodes, key=lambda n: n.year or 0)

    ranks = OrderedDict()
    for node in nodes:
        lst = ranks.get(node.rank, [])
        lst.append(node)
        ranks[node.rank] = lst
        
    graph = Digraph()
    # The timeline graph
    with graph.subgraph() as sub_graph:
        sub_graph.attr('node', shape='plaintext')
        years = [k for k in ranks.keys() if k > 1900]
        for iy in range(len(years)-1):
            sub_graph.edge(str(years[iy]), str(years[iy+1]))
        
    # The nodes
    for rank, lst in ranks.items():
        if not rank:
            for node in lst:
                node.add_to_graph(graph)
        else:
            with graph.subgraph() as sub_graph:
                sub_graph.attr(rank='same')
                if rank > 1900:
                    sub_graph.node(str(rank))
                for node in lst:
                    node.add_to_graph(sub_graph)
                
    for edge in edges:
        edge.add_to_graph(graph)
    
    graph.render(output, format=format)
    

def generate_md():
    md = """# RL Timeline and Classification

This is a loose timeline and classification of RL algorithms/agents/models mainly by its learning methods:
- value based
- policy based
- actor critic

Additional agent labels are:
"""
    for f in Flag:
        md += f'- {f.value} ({f.name})\n'
    md += '\n'
    
    md += '![RL Classification](rl.gv.svg "RL Classification")\n\n'

    collect_nodes()
    md += rl.to_md(True, 2)
    
    md += '<HR>\n(This document is autogenerated)\n'
    return md


if __name__ == '__main__':
    generate_dot('rl.gv')
    
    with open('README.md', 'wt') as f:
        f.write(generate_md())
