# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections import OrderedDict
import copy
from enum import Enum
import re
import sys

# "Orientation" of the graph: left-right or top-bottom.
# If top-bottom, the flags will be shown
RANKDIR = "LR"  # LR or TB

# Edge style for weak connection between nodes
WEAK_LINK = 'dashed'

# Sometimes we add connection between nodes just to maintain ordering.
# This is the style of the edge for such connection. 
INVIS = "invis"

USE_TIMELINE = True
FONT_NAME = "arial"
NODE_FONT_SIZE = 12
TIMELINE_FONT_SIZE = 10
EDGE_FONT_COLOR = "darkgray"
EDGE_FONT_SIZE = 10

# Useful graphviz links:
# - https://graphviz.readthedocs.io/en/stable/index.html
# - http://www.graphviz.org/pdf/dotguide.pdf
# - https://graphviz.org/doc/info/attrs.html


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
    ADV = "Advantage"
    # Miscellaneous:
    RB = "Replay Buffer"
    RNN = "Recurrent Neural Network"



def url2md(url):
    """
    Get Markdown of url. url can be string or (title,url) sequence
    """
    if isinstance(url, str):
        o = re.search('//([^/]+)/', url)
        if not o:
            name = 'link'
        else:
            parts = o.group(1).split('.')
            name = parts[-2] + '.' + parts[-1]
        return f'[{name}]({url})'
    else:
        return f'[{url[0]}]({url[1]})'
    

class Edge:
    """
    An Edge is a connection between Nodes/Groups
    """
    
    def __init__(self, dest, **attrs):
        self.dest = dest
        #self.label = label # to prevent label from being displayed in graph
        self.attrs = attrs
        
    @property
    def label(self):
        return self.attrs.get('label', '')

    @property
    def invisible(self):
        return self.attrs.get('style', '') == INVIS
    

class NodeBase(ABC):
    """
    Base class for Nodes and Groups.
    """
    def __init__(self, title, description, group, flags=[], authors=None, year=None, url=None,
                 videos=[], links=[], graph_type="node", output_md=True, **attrs):
        assert graph_type in ['node', 'cluster', 'node']
        self.title = title
        self.description = description
        self.group = group
        self.flags = flags
        self.authors = authors
        self.year = year
        self.url = url
        self.videos = videos
        self.links = links
        self.graph_type = graph_type
        self.output_md = output_md
        self.attrs = attrs
        if group:
            group.nodes.append(self)
        self.out_edges = []
        self.in_edges = []

    def __str__(self):
        return self.title
    
    @property
    def name(self):
        """
        Suitable name to be used as HTML anchor etc.
        """
        return re.sub('[^0-9a-zA-Z]+', '', self.title)

    @property
    def graph_name(self):
        """
        Identification of this node in the graph. If type is cluster, the name needs to be
        prefixed with "cluster" (graphviz convention)
        """
        return ('cluster' + self.title) if self.graph_type=='cluster' else self.title
        
    @property
    def graph_rank(self):
        """
        The rank of this node in the graph/cluster.
        """
        return self.year or 0
        
    def connect(self, other_node, **attrs):
        """
        Add connection from this node to other node. attrs are edge attributes.
        """
        self.out_edges.append( Edge(other_node, **attrs) )
        other_node.in_edges.append( Edge(self, **attrs) )
        
    def get_parents(self):
        """
        Get list of parents from root up to immediate parent.
        """
        p = self
        parents = []
        while p.group:
            parents.append(p.group)
            p = p.group
        parents.reverse()
        return parents

    @abstractmethod
    def collect_nodes(self):
        pass
    
    @abstractmethod
    def export_node(self, graph):
        pass

    @abstractmethod
    def export_connections(self, graph, cluster):
        pass

    @abstractmethod
    def export_graph(self, graph, cluster):
        pass

    @abstractmethod
    def export_md(self):
        pass
            
    def _export_node(self, graph):
        if self.graph_type != "node":
            return
        if RANKDIR=="LR":
            # For left to right rank
            attrs = copy.copy(self.attrs)
            attrs['fontname'] = FONT_NAME
            attrs['fontsize'] = str(NODE_FONT_SIZE)
            if 'shape' not in attrs:
                attrs['shape']='record'
            if 'style' not in attrs:
                attrs['style']='rounded'
            if not USE_TIMELINE:
                year = f'|{self.year}' if self.year else ''
            else:
                year = ''
            graph.node(self.graph_name, label=f'{{{self.title}{year}}}', **attrs)
        else:
            # For top down rank
            fields = []
            if self.year:
                fields.append(str(self.year))
            if self.flags:
                fields += [f.name for f in self.flags]
            label = self.title
            if fields:
                label += f'|{{{"|".join(fields)}}}'
            attrs = copy.copy(self.attrs)
            attrs['fontname'] = FONT_NAME
            attrs['fontsize'] = str(NODE_FONT_SIZE)
            if 'shape' not in attrs:
                attrs['shape']='record'
            if 'style' not in attrs:
                attrs['style']='rounded'
            graph.node(self.graph_name, label=f'{{{label}}}', **attrs)
        
    def _export_connections(self, graph, cluster):
        for edge in self.out_edges:
            attrs = copy.copy(edge.attrs)
            if attrs.get('style', '')==WEAK_LINK:
                attrs['color'] = 'darkgray'
                attrs['fontcolor'] = EDGE_FONT_COLOR
                attrs['fontsize'] = str(EDGE_FONT_SIZE)
                attrs['fontname'] = FONT_NAME
            if edge.dest.group == self.group:
                cluster.edge(self.graph_name, edge.dest.graph_name, **attrs)
            else:
                graph.edge(self.graph_name, edge.dest.graph_name, **attrs)
        
    def _export_md(self):
        if not self.output_md:
            return f' <a name="{self.name}"></a>\n'
        parents = self.get_parents()
        md = ('#' * min(len(parents)+2,5)) + f' <a name="{self.name}"></a>{self.title}\n'
 
        if parents:
            paths = parents + [self]
            md += f'(Path: '
            md += ' --> '.join([f'[{p.title}](#{p.name})' for p in paths]) + ')\n\n'
        if self.description:
            md += f'{self.description}\n\n'
        if self.url:
            md += f'- Paper: {self.url}\n'
        if self.authors:
            md += f'- Authors: {self.authors}\n'
        if self.year:
            md += f'- Year: {self.year}\n'
        if self.flags:
            md += '- Flags:\n'
            for f in self.flags:
                md += f'  - {f.value} ({f.name})\n'
        if self.in_edges:
            md += f'- Related to prior idea{"s" if len(self.in_edges)>1 else ""}:\n'
            for e in self.in_edges:
                if e.invisible:
                    continue
                md += f'  - [{e.dest.title}](#{e.dest.name})'
                if e.label:
                    md += f' ({e.label})'
                md += '\n'
        if self.out_edges:
            md += f'- Related to subsequent idea{"s" if len(self.out_edges)>1 else ""}:\n'
            for e in self.out_edges:
                if e.invisible:
                    continue
                md += f'  - [{e.dest.title}](#{e.dest.name})'
                if e.label:
                    md += f' ({e.label})'
                md += '\n'
        if self.links:
            md += '- Useful links:\n' + '\n'.join([f'  - {url2md(l)}' for l in self.links]) + '\n'
        if self.videos:
            md += '- Videos:\n' + '\n'.join([f'  - {url2md(l)}' for l in self.videos]) + '\n'
        md += '\n'
        
        return md


class Group(NodeBase):
    """
    Group is a "container" for other nodes.
    """
    
    def __init__(self, title, description, group, flags=[], authors=None, year=None, url=None,
                 videos=[], links=[], graph_type="cluster", timeline=False, output_md=True, same_rank=False, 
                 **attrs):
        super().__init__(title, description, group, flags=flags, authors=authors, year=year, url=url,
                 videos=videos, links=links, graph_type=graph_type, output_md=output_md, **attrs)
        self.nodes = []
        self.timeline = timeline
        self.same_rank = same_rank
        
    def collect_nodes(self):
        nodes = [self]
        for node in self.nodes:
            nodes += node.collect_nodes()
        return nodes
        
    def export_node(self, graph):
        self._export_node(graph)
        for child in self.nodes:
            child.export_node(graph)
            
    def export_connections(self, graph, cluster):
        if self.graph_type == "cluster":
            with cluster.subgraph(name=self.graph_name) as c:
                c.attr(label=self.title)
                c.attr(color='black')
                #c.node_attr['style'] = 'filled'
                for child in self.nodes:
                    child.export_connections(graph, c)
        else:
            for child in self.nodes:
                child.export_connections(graph, cluster)

        self._export_connections(graph, cluster)
    
    def _export_graph_with_timeline(self, graph, cluster):
        nodes = copy.copy(self.nodes)
        nodes = sorted(nodes, key=lambda n: n.graph_rank)
        
        ranks = OrderedDict()
        for node in nodes:
            lst = ranks.get(node.graph_rank, [])
            lst.append(node)
            ranks[node.graph_rank] = lst

        # The nodes
        for rank, lst in ranks.items():
            if not rank:
                for node in lst:
                    node.export_graph(graph, cluster)
            else:
                with cluster.subgraph() as rank_graph:
                    rank_graph.attr(rank='same')
                    if rank > 1900:
                        rank_graph.node(f'{self.title}{rank}', label=str(rank), fontcolor='darkgray',
                                        fontname=FONT_NAME, fontsize=str(TIMELINE_FONT_SIZE), 
                                        shape='plaintext', group=f'timeline{self.title}')

                    for node in lst:
                        node._export_node(rank_graph)
                        node._export_connections(graph, cluster)

        # The timeline graph
        with cluster.subgraph(name='clusterTimeline' + self.title) as timeline_graph:
            years = [k for k in ranks.keys() if k > 1900]
            for iy in range(len(years)-1):
                timeline_graph.edge(f'{self.title}{years[iy]}', f'{self.title}{years[iy+1]}',
                                    color='darkgray')

            
    def export_graph(self, graph, cluster):
        if self.graph_type == "cluster":
            with cluster.subgraph(name=self.graph_name) as c:
                c.attr(label=self.title)
                c.attr(color='black')
                c.attr(style='dashed')
                c.attr(fontname=FONT_NAME)
                c.attr(fontsize='16')
                if self.same_rank:
                    c.attr(rank='same')
                if self.timeline:
                    self._export_graph_with_timeline(graph, c)
                else:
                    self._export_node(c)
                    for child in self.nodes:
                        child.export_graph(graph, c)
        else:
            self._export_node(cluster)
            for child in self.nodes:
                child.export_graph(graph, cluster)
        self._export_connections(graph, cluster)

    def export_md(self):
        md = self._export_md()
        for child in self.nodes:
            md += child.export_md()
        return md
    

class Node(NodeBase):
    """
    A Node represents an algorithm. The relevant properties can be initialized from
    the constructor.
    """
    
    def __init__(self, title, description, group, flags=[], authors=None, year=None, url=None,
                 videos=[], links=[], graph_type="node", output_md=True, **attrs):
        super().__init__(title, description, group, flags=flags, authors=authors, year=year, url=url,
                 videos=videos, links=links, graph_type=graph_type, output_md=output_md, **attrs)
    
    def collect_nodes(self):
        return [self]
        
    def export_node(self, graph):
        self._export_node(graph)

    def export_connections(self, graph, cluster):
        self._export_connections(graph, cluster)

    def export_graph(self, graph, cluster):
        self._export_node(cluster)
        self._export_connections(graph, cluster)
        
    def export_md(self):
        return self._export_md()


#
# The nodes. Note: Within their group, keep nodes relatively sorted by their publication year
#
rl = Group('Reinforcement Learning', 
           'Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward [from Wikipedia]',
           None, graph_type="node", same_rank=True,
           links=[('A (Long) Peek into Reinforcement Learning', 'https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html'),
                  ('(book) Reinforcement Learning: An Introduction - 2nd Edition - Richard S. Sutton and Andrew G. Barto', 'http://incompleteideas.net/book/the-book.html')
               ],
           videos=[('(playlist ) Introduction to Reinforcement learning with David Silver', 'https://www.youtube.com/playlist?list=PLqYmG7hTraZBiG_XpjnPrSNw-1XQaM_gB'),
                   ('(playlist ) Reinforcement Learning Course | DeepMind & UCL', 'https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb'),
                   ('(playlist ) Reinforcement Learning Tutorials', 'https://www.youtube.com/playlist?list=PLWzQK00nc192L7UMJyTmLXaHa3KcO0wBT'),
                   ('(playlist ) Deep RL Bootcamp 2017', 'https://www.youtube.com/playlist?list=PLAdk-EyP1ND8MqJEJnSvaoUShrAWYe51U'),
                   ('(playlist ) CS885 Reinforcement Learning - Spring 2018 - University of Waterloo', 'https://www.youtube.com/playlist?list=PLdAoL1zKcqTXFJniO3Tqqn6xMBBL07EDc'),
                   ('(playlist ) CS234: Reinforcement Learning | Winter 2019', 'https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u'),
               ])

value_gradient = Group('Value Gradient', 
                    'The algorithm is learning the value function of each state or state-action. The policy is implicit, usually by just selecting the best value',
                    rl, timeline=USE_TIMELINE)

policy_gradient = Group('Policy Gradient/Actor-Critic', 
                     'The algorithm works directly to optimize the policy, with or without value function. If the value function is learned in addition to the policy, we would get Actor-Critic algorithm. Most policy gradient algorithms are Actor-Critic. The *Critic* updates value function parameters *w* and depending on the algorithm it could be action-value ***Q(a|s;w)*** or state-value ***V(s;w)***. The *Actor* updates policy parameters θ, in the direction suggested by the critic, ***π(a|s;θ)***. [from [Lilian Weng\' blog](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)]', 
                     rl, timeline=USE_TIMELINE,
                     links=[
                        ('Policy Gradient Algorithms', 'https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html'),
                        ('RL — Policy Gradient Explained', 'https://medium.com/@jonathan_hui/rl-policy-gradients-explained-9b13b688b146'),
                        ('Going Deeper Into Reinforcement Learning: Fundamentals of Policy Gradients', 'https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/'),
                        ('An introduction to Policy Gradients with Cartpole and Doom', 'https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/')
                        ])

root_value_gradient = Node('vg', '', value_gradient, output_md=False, style=INVIS)
root_policy_gradient = Node('pg', '', policy_gradient, output_md=False, style=INVIS)

rl.connect(root_value_gradient, lhead=value_gradient.graph_name)
rl.connect(root_policy_gradient, lhead=policy_gradient.graph_name)
 

#
# VALUE GRADIENT
#
sarsa = Node('SARSA',
             'SARSA (State-Action-Reward-State-Action) is an on-policy TD control method',
             value_gradient,
             flags=[Flag.ONP, Flag.DA],
             authors='G. A. Rummery, M. Niranjan',
             year=1994, 
             url='http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.2539&rep=rep1&type=pdf')
root_value_gradient.connect(sarsa, style=INVIS)

qlearning = Node('Q-learning',
           'Q-learning an off-policy TD control method. Unlike SARSA, it doesn\'t follow the policy to find the next action but rather chooses most optimal action in a greedy fashion',
           value_gradient,
           flags=[Flag.OFP, Flag.DA],
           authors='Chris Watkins',
           year=1989, 
           url='http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf',
           links=[('Diving deeper into Reinforcement Learning with Q-Learning', 'https://www.freecodecamp.org/news/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe/'),
                  ('Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks', 'https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0')]
           )
root_value_gradient.connect(qlearning, style=INVIS)

dqn = Node('DQN',
           'Deep Q Network. Q-Learning with using deep neural network as value estimator',
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB],
           authors='Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller',
           year=2013, 
           url='https://arxiv.org/abs/1312.5602',
           links=[('(tutorial) Deep Q Learning for the CartPole', 'https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f'),
                  ('An introduction to Deep Q-Learning: let’s play Doom', 'https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/')])
qlearning.connect(dqn)

drqn = Node('DRQN',
           'Deep Recurrent Q-Learning. Adding recurrency to a Deep Q-Network (DQN) by replacing the first post-convolutional fully-connected layer with a recurrent LSTM',
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB, Flag.RNN],
           authors='Matthew Hausknecht, Peter Stone',
           year=2015, 
           url='https://arxiv.org/abs/1507.06527',
           links=[])
dqn.connect(drqn)

ddqn = Node('DDQN',
            'Double DQN adds another neural network, making separate network for policy and target. The target network is only updated after certain number of steps/episodes. This makes the learning more stable.',
            value_gradient,
            flags=[Flag.OFP, Flag.CS, Flag.DA],
            authors='Hado van Hasselt, Arthur Guez, David Silver',
            year=2015,
            url='https://arxiv.org/abs/1509.06461',
            links=[('(tutorial) Deep Q Learning for the CartPole', 'https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f')])
dqn.connect(ddqn)

dqn_per = Node('PER',
           'Prioritized Experience Replay (PER) improves data efficiency by replaying transitions from which there is more to learn more often',
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB],
           authors='Tom Schaul, John Quan, Ioannis Antonoglou, David Silver',
           year=2015, 
           url='https://arxiv.org/abs/1511.05952',
           links=[])
dqn.connect(dqn_per)

duel_dqn = Node('Duelling-DQN',
                'Duelling DQN represents two separate estimators: one for the state value function and one for the state-dependent action advantage function. The main benefit of this factoring is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm.',
                value_gradient,
                flags=[Flag.OFP, Flag.CS, Flag.DA],
                authors='Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas',
                year=2016, 
                url='https://arxiv.org/abs/1511.06581')
ddqn.connect(duel_dqn)

qr_dqn = Node('QR-DQN',
           'Distributional Reinforcement Learning with Quantile Regression (QR-DQN). In QR-DQN, distribution of values values are used for each state-action pair instead of a single mean value',
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB],
           authors='Will Dabney, Mark Rowland, Marc G. Bellemare, Rémi Munos',
           year=2017, 
           url='https://arxiv.org/abs/1710.10044',
           links=[('(GitHub) Quantile Regression DQN', 'https://github.com/senya-ashukha/quantile-regression-dqn-pytorch')])
dqn.connect(qr_dqn)

c51 = Node('C51',
           'C51 Algorithm. The core idea of Distributional Bellman is to ask the following questions. If we can model the Distribution of the total future rewards, why restrict ourselves to the expected value (i.e. Q function)? There are several benefits to learning an approximate distribution rather than its approximate expectation. [[source: flyyufelix\'s blog](https://flyyufelix.github.io/2017/10/24/distributional-bellman.html)]',
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB],
           authors='Marc G. Bellemare, Will Dabney, Rémi Munos',
           year=2017, 
           url='https://arxiv.org/abs/1707.06887',
           links=[('Distributional Bellman and the C51 Algorithm', 'https://flyyufelix.github.io/2017/10/24/distributional-bellman.html')])
root_value_gradient.connect(c51, style=INVIS)
#dqn_per.connecT(c51, syle=INVIS)

rainbow = Node('RAINBOW',
           'Examines six extensions to the DQN algorithm and empirically studies their combination',
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB],
           authors='Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver',
           year=2017, 
           url='https://arxiv.org/abs/1710.02298',
           links=[])
ddqn.connect(rainbow, style=WEAK_LINK)
dqn_per.connect(rainbow, style=WEAK_LINK)
duel_dqn.connect(rainbow, style=WEAK_LINK)
qr_dqn.connect(rainbow, style=WEAK_LINK)

dqn_her = Node('DQN+HER',
           'DQN with Hindsight Experience Replay (HER)',
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB],
           authors='Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, Wojciech Zaremba',
           year=2017, 
           url='https://arxiv.org/abs/1707.01495',
           links=[('Learning from mistakes with Hindsight Experience Replay', 'https://becominghuman.ai/learning-from-mistakes-with-hindsight-experience-replay-547fce2b3305')])
dqn.connect(dqn_her)


#
# POLICY GRADIENT / ACTOR-CRITIC
#
reinforce = Node('REINFORCE',
           'REINFORCE (Monte-Carlo policy gradient) is a pure policy gradient algorithm that works without a value function. The agent collects a trajectory of one episode using its current policy, and uses the returns to update the policy parameter',
           policy_gradient,
           flags=[Flag.MC, Flag.ONP, Flag.CS, Flag.DA],
           authors='Ronald J. Williams',
           year=1992, 
           url='https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf',
           links=[('LearningReinforcementLearningbyLearningREINFORCE (PDF)', 'http://www.cs.toronto.edu/~tingwuwang/REINFORCE.pdf'), 
                  ('An introduction to Policy Gradients with Cartpole and Doom', 'https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/')
                  ]
           )
root_policy_gradient.connect(reinforce, style=INVIS)

"""
vpg = Node('VPG',
           'Vanilla Policy Gradient',
           policy_gradient,
           flags=[Flag.MC, Flag.ONP, Flag.CS, Flag.DA, Flag.CA],
           authors='Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour',
           year=2000, 
           url='https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf',
           links=['https://spinningup.openai.com/en/latest/algorithms/vpg.html']
           )
"""

dpg = Node('DPG',
           'Deterministic Policy Gradient. Abstract: In this paper we consider deterministic policy gradient algorithms for reinforcement learning with continuous actions. The deterministic policy gradient has a particularly appealing form: it is the expected gradient of the action-value function. This simple form means that the deterministic policy gradient can be estimated much more efficiently than the usual stochastic policy gradient. To ensure adequate exploration, we introduce an off-policy actor-critic algorithm that learns a deterministic target policy from an exploratory behaviour policy. We demonstrate that deterministic policy gradient algorithms can significantly outperform their stochastic counterparts in high-dimensional action spaces.',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.DP],
           authors='David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller',
           year=2014, 
           url='http://proceedings.mlr.press/v32/silver14.pdf',
           links=[]
           )
root_policy_gradient.connect(dpg, style=INVIS)

ddpg = Node('DDPG',
           'Deep Deterministic Policy Gradient (DDPG).',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.DP, Flag.RB],
           authors='Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra',
           year=2015, 
           url='https://arxiv.org/abs/1509.02971',
           links=[('Deep Deterministic Policy Gradient - Spinning Up', 'https://spinningup.openai.com/en/latest/algorithms/ddpg.html')
                  ]
           )
dpg.connect(ddpg)
dqn.connect(ddpg, style=WEAK_LINK, label='replay buffer', constraint="false")

trpo = Node('TRPO',
           'Trust Region Policy Optimization (TRPO) improves training stability by enforcing a KL divergence constraint to avoid parameter updates that change the policy too much at one step.',
           policy_gradient,
           flags=[Flag.ONP, Flag.CS, Flag.CA, Flag.ADV],
           authors='John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel',
           year=2015, 
           url='https://arxiv.org/pdf/1502.05477',
           links=[('RL — Trust Region Policy Optimization (TRPO) Explained', 'https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9'),
                  ('RL — Trust Region Policy Optimization (TRPO) Part 2', 'https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-part-2-f51e3b2e373a')]
           )
root_policy_gradient.connect(trpo, style=INVIS)

gae = Node('GAE',
           'Generalized Advantage Estimation',
           policy_gradient,
           flags=[Flag.ONP, Flag.CS, Flag.CA],
           authors='John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel',
           year=2015, 
           url='https://arxiv.org/abs/1506.02438',
           links=[('Generalized Advantage Estimator Explained','https://notanymike.github.io/GAE/'),
                  ('Notes on the Generalized Advantage Estimation Paper', 'https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/')]
           )
root_policy_gradient.connect(gae, style=INVIS)
trpo.connect(gae, style=WEAK_LINK)

a3c = Node('A3C',
           'Asynchronous Advantage Actor-Critic (A3C) is a classic policy gradient method with the special focus on parallel training. In A3C, the critics learn the state-value function, ***V(s;w)***, while multiple actors are trained in parallel and get synced with global parameters from time to time. Hence, A3C is good for parallel training by default, i.e. on one machine with multi-core CPU. [from [Lilian Weng\' blog](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)]',
           policy_gradient,
           flags=[Flag.ONP, Flag.CS, Flag.CA, Flag.ADV, Flag.SP],
           authors='Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu',
           year=2016, 
           url='https://arxiv.org/abs/1602.01783',
           links=[('Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C)', 'https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2'),
                  ('An implementation of A3C', 'https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c')]
           )
root_policy_gradient.connect(a3c, style=INVIS)
a3c.connect(rainbow, style=WEAK_LINK)

ddpg_her = Node('DDPG+HER',
           'Hindsight Experience Replay (HER)',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.DP, Flag.RB],
           authors='Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, Wojciech Zaremba',
           year=2017, 
           url='https://arxiv.org/abs/1707.01495',
           links=['https://becominghuman.ai/learning-from-mistakes-with-hindsight-experience-replay-547fce2b3305'])
ddpg.connect(ddpg_her, style=WEAK_LINK)

maddpg = Node('MADDPG',
           'Multi-agent DDPG (MADDPG) extends DDPG to an environment where multiple agents are coordinating to complete tasks with only local information. In the viewpoint of one agent, the environment is non-stationary as policies of other agents are quickly upgraded and remain unknown. MADDPG is an actor-critic model redesigned particularly for handling such a changing environment and interactions between agents (from [Lilian Weng\'s blog](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#maddpg))',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.DP, Flag.RB],
           authors='Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, Igor Mordatch',
           year=2017, 
           url='https://arxiv.org/abs/1706.02275',
           links=[]
           )
ddpg.connect(maddpg)

a2c = Node('A2C',
           'A2C is a synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C). It uses multiple workers to avoid the use of a replay buffer.',
           policy_gradient,
           flags=[Flag.ONP, Flag.CS, Flag.CA, Flag.ADV, Flag.SP],
           authors='OpenAI',
           year=2017, 
           url='https://openai.com/blog/baselines-acktr-a2c/',
           links=[
               ('OpenAI Baselines: ACKTR & A2C', 'https://openai.com/blog/baselines-acktr-a2c/'),
               ('An intro to Advantage Actor Critic methods: let’s play Sonic the Hedgehog!', 'https://www.freecodecamp.org/news/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d/'),
               ('Stable Baselines: A2C', 'https://stable-baselines.readthedocs.io/en/master/modules/a2c.html')
               ]
           )
a3c.connect(a2c)

acer = Node('ACER',
           'Actor-Critic with Experience Replay (ACER) combines several ideas of previous algorithms: it uses multiple workers (as A2C), implements a replay buffer (as in DQN), uses Retrace for Q-value estimation, importance sampling and a trust region. ACER is A3C\'s off-policy counterpart. ACER proposes several designs to overcome the major obstacle to making A3C off policy, that is how to control the stability of the off-policy estimator. (source: [Lilian Weng\'s blog](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#acer))',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.ADV, Flag.RB],
           authors='Ziyu Wang, Victor Bapst, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, Nando de Freitas',
           year=2017, 
           url='https://arxiv.org/abs/1611.01224',
           links=[
               ]
           )
a3c.connect(acer)
dqn.connect(acer, style=WEAK_LINK, label='replay buffer, workers')
#a2c.connect(acer, style=WEAK_LINK, label='multiple workers')
a2c.connect(acer, style=INVIS)
trpo.connect(acer, style=WEAK_LINK, label='TRPO technique')

acktr = Node('ACKTR',
           'Actor Critic using Kronecker-Factored Trust Region (ACKTR) is applying trust region optimization to deep reinforcement learning using a recently proposed Kronecker-factored approximation to the curvature.',
           policy_gradient,
           flags=[Flag.ONP, Flag.CS, Flag.CA, Flag.ADV],
           authors='Yuhuai Wu, Elman Mansimov, Shun Liao, Roger Grosse, Jimmy Ba',
           year=2017, 
           url='https://arxiv.org/abs/1708.05144',
           links=[
               ]
           )
root_policy_gradient.connect(acktr, style=INVIS)
a2c.connect(acktr, style=INVIS) # just to maintain relative timeline order

ppo = Node('PPO',
           'Proximal Policy Optimization (PPO) is similar to [TRPO](#TRPO) but uses simpler mechanism while retaining similar performance.',
           policy_gradient,
           flags=[Flag.ONP, Flag.CS, Flag.DA, Flag.CA, Flag.ADV],
           authors='John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov',
           year=2017, 
           url='https://arxiv.org/abs/1707.06347',
           links=['https://spinningup.openai.com/en/latest/algorithms/ppo.html',
                  'https://openai.com/blog/openai-baselines-ppo/'],
           videos=[('Policy Gradient methods and Proximal Policy Optimization (PPO): diving into Deep RL!', 'https://www.youtube.com/watch?v=5P7I-xPq8u8')]
           )
trpo.connect(ppo, style=WEAK_LINK)

svpg = Node('SVPG',
           'Stein Variational Policy Gradient (SVPG)',
           policy_gradient,
           flags=[Flag.ONP, Flag.CS, Flag.DA, Flag.CA],
           authors='Yang Liu, Prajit Ramachandran, Qiang Liu, Jian Peng',
           year=2017, 
           url='https://arxiv.org/abs/1704.02399',
           links=[('Policy Gradient Algorithms', 'https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#svpg'),
                  ]
           )
root_policy_gradient.connect(svpg, style=INVIS)
a2c.connect(svpg, style=INVIS) # just to maintain relative timeline order

d4pg = Node('D4PG',
           'Distributed Distributional Deep Deterministic Policy Gradient (D4PG) adopts the very successful distributional perspective on reinforcement learning and adapts it to the continuous control setting. It combines this within a distributed framework. It also combines this technique with a number of additional, simple improvements such as the use of N-step returns and prioritized experience replay [from the paper\'s abstract]',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.DP, Flag.RB],
           authors='Gabriel Barth-Maron, Matthew W. Hoffman, David Budden, Will Dabney, Dan Horgan, Dhruva TB, Alistair Muldal, Nicolas Heess, Timothy Lillicrap',
           year=2018, 
           url='https://arxiv.org/abs/1804.08617',
           links=[]
           )
ddpg.connect(d4pg)

sac = Node('SAC',
           'Soft Actor Critic (SAC) is an algorithm that optimizes a stochastic policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches.',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.CA, Flag.SP],
           authors='Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine',
           year=2018, 
           url='https://arxiv.org/abs/1801.01290',
           links=[('Spinning Up SAC page', 'https://spinningup.openai.com/en/latest/algorithms/sac.html'),
                  ('(GitHub) SAC code by its author', 'https://github.com/haarnoja/sac')])
root_policy_gradient.connect(sac, style=INVIS)
ppo.connect(sac, style=INVIS) # just to maintain relative timeline order

td3 = Node('TD3',
           'Twin Delayed DDPG (TD3). TD3 addresses function approximation error in DDPG by introducing twin Q-value approximation network and less frequent updates',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.DP, Flag.RB],
           authors='Scott Fujimoto, Herke van Hoof, David Meger',
           year=2018, 
           url='https://arxiv.org/abs/1802.09477',
           links=[('Twin Delayed DDPG (Spinning Up)', 'https://spinningup.openai.com/en/latest/algorithms/td3.html')])
ddpg.connect(td3)
ddqn.connect(td3, style=WEAK_LINK, label='double Q-learning')

impala = Node('IMPALA',
           'Importance Weighted Actor-Learner Architecture (IMPALA)',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.CA],
           authors='Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu',
           year=2018, 
           url='https://arxiv.org/abs/1802.01561',
           links=[('Policy Gradient Algorithms', 'https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html')])
root_policy_gradient.connect(impala, style=INVIS)
a2c.connect(impala, style=INVIS) # just to maintain relative timeline order


def generate_graph(output, format, use_rank=True):
    from graphviz import Digraph, Source

    graph = Digraph()
    graph.attr(compound='true')
    graph.attr(rankdir=RANKDIR)
    graph.attr(rank='same') 
    
    """
    ranks = OrderedDict()
    nodes = rl.collect_nodes()
    nodes = sorted(nodes, key=lambda n: n.graph_rank)
    for node in nodes:
        lst = ranks.get(node.graph_rank, [])
        lst.append(node)
        ranks[node.graph_rank] = lst
    
    # The timeline graph
    with graph.subgraph(name='clusterTimeline') as timeline_graph:
        #timeline_graph.attr(rankdir=RANKDIR)
        timeline_graph.attr('node', shape='plaintext')
        years = [k for k in ranks.keys() if k > 1900]
        for iy in range(len(years)-1):
            timeline_graph.edge(str(years[iy]), str(years[iy+1]))
    """
    
    rl.export_graph(graph, graph)
    graph.render(output, format=format)


def generate_md():
    nodes = rl.collect_nodes()
    
    md = """# RL Taxonomy

This is a loose taxonomy of reinforcement learning algorithms. I'm by no means expert in this area, I'm making this as part of my learning process, so please PR to correct things or suggest new stuff.

"""
    md += '#### Table of Contents:<HR>\n\n'
    md += "[Taxonomy](#taxonomy)<BR>\n"
    for node in nodes:
        if not node.output_md:
            continue
        parents = node.get_parents()
        if len(parents):
            md += '  ' * (len(parents)-1)
        if parents:
            md += '- '
        md += f'[{node.title}](#{node.name})\n'
    md += '\n'

    md += """## <A name="taxonomy"></a>Taxonomy

Solid line indicates some progression from one idea to another. Dashed line indicates a loose connection, which could be as little as mentioning of the idea in the newer paper.

![RL Taxonomy](rl-taxonomy.gv.svg "RL Taxonomy")\n\n"""

    if RANKDIR != 'LR':
        md += 'Note: labels attached to algorithms are:\n'
        for f in Flag:
            md += f'- {f.value} ({f.name})\n'
        md += '\n'

    md += rl.export_md()

    md += """<HR>
    
Sources:
- [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
- [Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)

"""
        
    md += '\n(This document is autogenerated)\n'
    return md


if __name__ == '__main__':
    generate_graph('rl-taxonomy.gv', 'svg', use_rank=False)
    
    with open('README.md', 'w', encoding="utf-8") as f:
        f.write(generate_md())
