# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections import OrderedDict
import copy
from enum import Enum
import re
import sys

import html2text
import markdown


# "Orientation" of the graph: left-right or top-bottom.
RANKDIR = "LR"  # LR or TB

# Edge style for weak connection between nodes
WEAK_LINK = 'dashed'

# Sometimes we add connection between nodes just to maintain ordering.
# This is the style of the edge for such connection.
INVIS = "invis"

# Style from rood node
ROOT_EDGE = "solid"

# Style for ordering edge
ORDER_EDGE = "invis"

FONT_NAME = "sans-serif"
CLUSTER_FONT_NAME = "arial black"
NODE_FONT_NAME = "helvetica-bold"
NODE_FONT_SIZE = 12
TIMELINE_FONT_NAME = NODE_FONT_NAME
TIMELINE_COLOR = "white"
TIMELINE_FONT_SIZE = 14
TIMELINE_FILL_COLOR = '#707070'
EDGE_FONT_COLOR = 'black'
EDGE_FONT_SIZE = 10

# Useful graphviz links:
# - https://graphviz.readthedocs.io/en/stable/index.html
# - http://www.graphviz.org/pdf/dotguide.pdf
# - https://graphviz.org/doc/info/attrs.html


class Flag(Enum):
    """
    These are various flags that can be attributed to an algorithm
    """

    # MF = "Model-Free" # is the default for all algorithms
    MB = "Model-Based"
    MC = "Monte Carlo"
    # TD = "Temporal Difference" # is the default for all algorithms
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
    DI = "Distributional"
    MG = "Model is Given"  # model-based flag
    ML = "Model is Learnt"  # model based flag


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


def md2txt(md):
    html = markdown.markdown(md)
    cvt = html2text.HTML2Text()
    cvt.ignore_emphasis = True
    cvt.ignore_links = True
    txt = cvt.handle(html)
    return txt
            
                
class Edge:
    """
    An Edge is a connection between Nodes/Groups
    """

    def __init__(self, dest, **attrs):
        self.dest = dest
        # self.label = label # to prevent label from being displayed in graph
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
    def md_title(self):
        """Normalized title for markdown"""
        return self.title.replace('\\n', ' ')
    
    @property
    def name(self):
        """
        Suitable name to be used as HTML anchor etc.
        """
        return re.sub('[^0-9a-zA-Z]+', '', self.md_title)

    @property
    def graph_name(self):
        """
        Identification of this node in the graph. If type is cluster, the name needs to be
        prefixed with "cluster" (graphviz convention)
        """
        return ('cluster' + self.title) if self.graph_type == 'cluster' else self.title

    @property
    def tooltip(self):
        lines = self.description.split('\n')
        if lines:
            return md2txt(lines[0]) + (f'({self.year})' if self.year else '')            
        else:
            return ''
        
    @property
    def graph_rank(self):
        """
        The rank of this node in the graph/cluster.
        """
        if self.year:
            if self.year >= 1980 and self.year < 2000:
                return '1980-90s'
            elif self.year >= 2000 and self.year < 2010:
                return '2000s'
            elif self.year >= 2010 and self.year <= 2015:
                return '2010-2015'
            else:
                return str(self.year)
        else:
            return ''

    def connect(self, other_node, **attrs):
        """
        Add connection from this node to other node. attrs are edge attributes.
        """
        self.out_edges.append(Edge(other_node, **attrs))
        other_node.in_edges.append(Edge(self, **attrs))

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
        attrs = copy.copy(self.attrs)
        attrs['fontname'] = NODE_FONT_NAME
        if 'fontsize' not in attrs:
            attrs['fontsize'] = str(NODE_FONT_SIZE)
        if 'shape' not in attrs:
            attrs['shape'] = 'box'
        if 'style' not in attrs:
            attrs['style'] = 'rounded,bold,filled'
        if 'fillcolor' not in attrs:
            attrs['fillcolor'] = '#dae8fc'
        if 'tooltip' not in attrs:
            attrs['tooltip'] = self.tooltip
        if True:
            #url = self.url
            url = f'https://github.com/bennylp/RL-Taxonomy#{self.name}'
            url = url.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")
            attrs['URL'] = url
        graph.node(self.graph_name, label=f'{self.title}', **attrs)

    def _export_connections(self, graph, cluster):
        for edge in self.out_edges:
            attrs = copy.copy(edge.attrs)
            attrs['fontcolor'] = EDGE_FONT_COLOR
            attrs['fontsize'] = str(EDGE_FONT_SIZE)
            attrs['fontname'] = FONT_NAME
            if attrs.get('style', '') == WEAK_LINK:
                attrs['color'] = 'darkgray'
                attrs['fontcolor'] = 'darkgray'
            if edge.dest.group == self.group:
                cluster.edge(self.graph_name, edge.dest.graph_name, **attrs)
            else:
                graph.edge(self.graph_name, edge.dest.graph_name, **attrs)

    def _export_md(self):
        if not self.output_md:
            return f' <a name="{self.name}"></a>\n'
        parents = self.get_parents()
        md = ('#' * min(len(parents) + 2, 5)) + f' <a name="{self.name}"></a>{self.md_title}\n'

        if parents:
            paths = parents + [self]
            md += f'(Path: '
            md += ' --> '.join([f'[{p.md_title}](#{p.name})' for p in paths]) + ')\n\n'
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
                md += f'  - [{e.dest.md_title}](#{e.dest.name})'
                if e.label:
                    md += f' ({e.label})'
                md += '\n'
        if self.out_edges:
            md += f'- Related to subsequent idea{"s" if len(self.out_edges)>1 else ""}:\n'
            for e in self.out_edges:
                if e.invisible:
                    continue
                md += f'  - [{e.dest.md_title}](#{e.dest.name})'
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
                 videos=[], links=[], graph_type="cluster", output_md=True,
                 **attrs):
        super().__init__(title, description, group, flags=flags, authors=authors, year=year, url=url,
                 videos=videos, links=links, graph_type=graph_type, output_md=output_md, **attrs)
        self.nodes = []

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
                # c.node_attr['style'] = 'filled'
                for child in self.nodes:
                    child.export_connections(graph, c)
        else:
            for child in self.nodes:
                child.export_connections(graph, cluster)

        self._export_connections(graph, cluster)

    def export_graph(self, graph, cluster):
        if self.graph_type == "cluster":
            with cluster.subgraph(name=self.graph_name) as c:
                c.attr(label=self.title)
                c.attr(color='black')
                if 'style' not in self.attrs:
                    c.attr(style='dashed')
                c.attr(fontname=CLUSTER_FONT_NAME)
                c.attr(fontsize='16')
                #c.attr(tooltip=self.tooltip) # probably not. tooltip for large area is confusing
                c.attr(**self.attrs)
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
rl = Group('Reinforcement\\nLearning',
           'Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward [from [Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)]',
           None, graph_type="node", shape='plaintext', style='', fontsize='18',  
           links=[('A (Long) Peek into Reinforcement Learning', 'https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html'),
                  ('(book) Reinforcement Learning: An Introduction - 2nd Edition - Richard S. Sutton and Andrew G. Barto', 'http://incompleteideas.net/book/the-book.html')
               ],
           videos=[('(playlist) Introduction to Reinforcement learning with David Silver', 'https://www.youtube.com/playlist?list=PLqYmG7hTraZBiG_XpjnPrSNw-1XQaM_gB'),
                   ('(playlist) Reinforcement Learning Course | DeepMind & UCL', 'https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb'),
                   ('(playlist) Reinforcement Learning Tutorials', 'https://www.youtube.com/playlist?list=PLWzQK00nc192L7UMJyTmLXaHa3KcO0wBT'),
                   ('(playlist) Deep RL Bootcamp 2017', 'https://www.youtube.com/playlist?list=PLAdk-EyP1ND8MqJEJnSvaoUShrAWYe51U'),
                   ('(playlist) CS885 Reinforcement Learning - Spring 2018 - University of Waterloo', 'https://www.youtube.com/playlist?list=PLdAoL1zKcqTXFJniO3Tqqn6xMBBL07EDc'),
                   ('(playlist) CS234: Reinforcement Learning | Winter 2019', 'https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u'),
               ])

model_free = Group('Model Free',
                    'In model free reinforcement learning, the agent directly tries to predict the value/policy without having or trying to model the environment',
                    rl, style='rounded,filled', fillcolor='#f7fdff')
root_model_free = Node('Model Free', model_free.description, model_free, output_md=False, fillcolor='#ffe6cc', weight='10')
rl.connect(root_model_free)

model_based = Group('Model Based',
                    'In model-based reinforcement learning, the agent uses the experience to try to model the environment, and then uses the model to predict the value/policy',
                    rl, style='rounded,filled', fillcolor='#dafdda',
                    links=[('Model-Based Reinforcement Learning: Theory and Practice', 'https://bair.berkeley.edu/blog/2019/12/12/mbpo/'),
                        ])
root_model_based = Node('Model Based', model_based.description, model_based, output_md=False, fillcolor='#ffe6cc')
rl.connect(root_model_based)

meta_rl = Group('Meta-RL',
                'In meta reinforcement learning, the agent is trained over distribution of tasks, and with the knowledge it tries to solve new unseen but related task.',
                rl, style='rounded,filled', fillcolor='#f5f5da',
                links=[('Meta Reinforcement Learning', 'https://lilianweng.github.io/lil-log/2019/06/23/meta-reinforcement-learning.html')
                       ])
root_meta_rl = Node('Meta-RL', meta_rl.description, meta_rl, year=2001, output_md=False, fillcolor='#ffe6cc')
rl.connect(root_meta_rl)

value_gradient = Group('Value Gradient',
                    'The algorithm is learning the value function of each state or state-action. The policy is implicit, usually by just selecting the best value',
                    model_free, style='rounded,dashed,filled', fillcolor='#daf0f6')

policy_gradient = Group('Policy Gradient/Actor-Critic',
                     'The algorithm works directly to optimize the policy, with or without value function. If the value function is learned in addition to the policy, we would get Actor-Critic algorithm. Most policy gradient algorithms are Actor-Critic. The *Critic* updates value function parameters *w* and depending on the algorithm it could be action-value ***Q(a|s;w)*** or state-value ***V(s;w)***. The *Actor* updates policy parameters θ, in the direction suggested by the critic, ***π(a|s;θ)***. [from [Lilian Weng\' blog](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)]',
                     model_free, style='rounded,dashed,filled', fillcolor='#daf0f6',
                     links=[
                        ('Policy Gradient Algorithms', 'https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html'),
                        ('RL — Policy Gradient Explained', 'https://medium.com/@jonathan_hui/rl-policy-gradients-explained-9b13b688b146'),
                        ('Going Deeper Into Reinforcement Learning: Fundamentals of Policy Gradients', 'https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/'),
                        ('An introduction to Policy Gradients with Cartpole and Doom', 'https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/')
                        ])

root_value_gradient = Node('Value Gradient', value_gradient.description, value_gradient, output_md=False, fillcolor='#ffe6cc')
root_policy_gradient = Node('Policy Gradient\\n/Actor-Critic', policy_gradient.description, policy_gradient, output_md=False, fillcolor='#ffe6cc')

root_model_free.connect(root_value_gradient)
root_model_free.connect(root_policy_gradient)

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
root_value_gradient.connect(sarsa, style=ROOT_EDGE)

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
root_value_gradient.connect(qlearning, style=ROOT_EDGE)

td_gammon = Node('TD-Gammon',
           'TD-Gammon is a model-free reinforcement learning algorithm similar to Q-learning, and uses a multi-layer perceptron with one hidden layer as the value function approximator. It learns the game entirely by playing against itself and achieves superhuman level of play.',
           value_gradient,
           flags=[],
           authors='Gerald Tesauro',
           year=1995,
           url='https://dl.acm.org/doi/10.1145/203330.203343',
           links=[]
           )
root_value_gradient.connect(td_gammon, style=ROOT_EDGE)

dqn = Node('DQN',
           'Deep Q Network (DQN) is Q-Learning with deep neural network as state-action value estimator and uses a replay buffer to sample experiences from previous trajectories to make learning more stable.',
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
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB, Flag.DI],
           authors='Will Dabney, Mark Rowland, Marc G. Bellemare, Rémi Munos',
           year=2017,
           url='https://arxiv.org/abs/1710.10044',
           links=[('(GitHub) Quantile Regression DQN', 'https://github.com/senya-ashukha/quantile-regression-dqn-pytorch')])
dqn.connect(qr_dqn)

c51 = Node('C51',
           'C51 Algorithm. The core idea of Distributional Bellman is to ask the following questions. If we can model the Distribution of the total future rewards, why restrict ourselves to the expected value (i.e. Q function)? There are several benefits to learning an approximate distribution rather than its approximate expectation. [[source: flyyufelix\'s blog](https://flyyufelix.github.io/2017/10/24/distributional-bellman.html)]',
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB, Flag.DI],
           authors='Marc G. Bellemare, Will Dabney, Rémi Munos',
           year=2017,
           url='https://arxiv.org/abs/1707.06887',
           links=[('Distributional Bellman and the C51 Algorithm', 'https://flyyufelix.github.io/2017/10/24/distributional-bellman.html')])
dqn.connect(c51)
# dqn_per.connecT(c51, syle=INVIS)

rainbow = Node('RAINBOW',
           'Combines six DQN extensions, namely Double Q-Learning, prioritized replay, dueling networks, multi-step learning, distributional DQN, and noisy DQN into single model to achieve state of the art performance',
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

iqn = Node('IQN',
           """Implicit Quantile Networks (IQN). From the abstract: In this work, we build on recent advances in distributional reinforcement learning to give a generally applicable, flexible, and state-of-the-art distributional variant of DQN. We achieve this by using quantile regression to approximate the full quantile function for the state-action return distribution. By reparameterizing a distribution over the sample space, this yields an implicitly defined return distribution and gives rise to a large class of risk-sensitive policies. We demonstrate improved performance on the 57 Atari 2600 games in the ALE, and use our algorithm's implicitly defined distributions to study the effects of risk-sensitive policies in Atari games. 
           """,
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB, Flag.DI],
           authors='Will Dabney, Georg Ostrovski, David Silver, Rémi Munos',
           year=2018,
           url='https://arxiv.org/abs/1806.06923',
           links=[('(StackExchange) How does Implicit Quantile-Regression Network (IQN) differ from QR-DQN?', 'https://datascience.stackexchange.com/questions/40874/how-does-implicit-quantile-regression-network-iqn-differ-from-qr-dqn')])
dqn.connect(iqn)
# dqn_per.connecT(c51, syle=INVIS)

apex_dqn = Node('APE-X DQN',
           'DQN with Distributed Prioritized Experience Replay',
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB],
           authors='Dan Horgan, John Quan, David Budden, Gabriel Barth-Maron, Matteo Hessel, Hado van Hasselt, David Silver',
           year=2018,
           url='https://arxiv.org/abs/1803.00933',
           links=[('Understanding and Implementing Distributed Prioritized Experience Replay (Horgan et al., 2018)', 'https://towardsdatascience.com/understanding-and-implementing-distributed-prioritized-experience-replay-horgan-et-al-2018-d2c1640e0520')])
dqn.connect(apex_dqn)
# dqn_per.connecT(c51, syle=INVIS)

r2d2 = Node('R2D2',
           """Recurrent Replay Distributed DQN (R2D2). (from the abstract) Building on the recent successes of distributed training of RL agents, in this paper we investigate the training of RNN-based RL agents from distributed prioritized experience replay. We study the effects of parameter lag resulting in representational drift and recurrent state staleness and empirically derive an improved training strategy. Using a single network architecture and fixed set of hyper-parameters, the resulting agent, Recurrent Replay Distributed DQN, quadruples the previous state of the art on Atari-57, and matches the state of the art on DMLab-30. It is the first agent to exceed human-level performance in 52 of the 57 Atari games.""",
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB],
           authors='Steven Kapturowski, Georg Ostrovski, John Quan, Remi Munos, Will Dabney',
           year=2019,
           url='https://openreview.net/forum?id=r1lyTjAqYX',
           links=[])
dqn.connect(r2d2)

ngu = Node('NGU',
           'Never Give Up (NGU). (from the abstract) We propose a reinforcement learning agent to solve hard exploration games by learning a range of directed exploratory policies. We construct an episodic memory-based intrinsic reward using k-nearest neighbors over the agent\'s recent experience to train the directed exploratory policies, thereby encouraging the agent to repeatedly revisit all states in its environment. A self-supervised inverse dynamics model is used to train the embeddings of the nearest neighbour lookup, biasing the novelty signal towards what the agent can control. We employ the framework of Universal Value Function Approximators (UVFA) to simultaneously learn many directed exploration policies with the same neural network, with different trade-offs between exploration and exploitation. By using the same neural network for different degrees of exploration/exploitation, transfer is demonstrated from predominantly exploratory policies yielding effective exploitative policies. The proposed method can be incorporated to run with modern distributed RL agents that collect large amounts of experience from many actors running in parallel on separate environment instances. Our method doubles the performance of the base agent in all hard exploration in the Atari-57 suite while maintaining a very high score across the remaining games, obtaining a median human normalised score of 1344.0%. Notably, the proposed method is the first algorithm to achieve non-zero rewards (with a mean score of 8,400) in the game of Pitfall! without using demonstrations or hand-crafted features.',
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB],
           authors='Adrià Puigdomènech Badia, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Bilal Piot, Steven Kapturowski, Olivier Tieleman, Martín Arjovsky, Alexander Pritzel, Andew Bolt, Charles Blundell',
           year=2020,
           url='https://arxiv.org/abs/2002.06038',
           links=[])
r2d2.connect(ngu)

agent57 = Node('Agent57',
           '(from the abstract) Atari games have been a long-standing benchmark in the reinforcement learning (RL) community for the past decade. This benchmark was proposed to test general competency of RL algorithms. Previous work has achieved good average performance by doing outstandingly well on many games of the set, but very poorly in several of the most challenging games. We propose Agent57, the first deep RL agent that outperforms the standard human benchmark on all 57 Atari games. To achieve this result, we train a neural network which parameterizes a family of policies ranging from very exploratory to purely exploitative. We propose an adaptive mechanism to choose which policy to prioritize throughout the training process. Additionally, we utilize a novel parameterization of the architecture that allows for more consistent and stable learning.',
           value_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB],
           authors='Adrià Puigdomènech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Charles Blundell',
           year=2020,
           url='https://arxiv.org/abs/2003.13350',
           links=[('DeepMind Unveils Agent57, the First AI Agents that Outperforms Human Benchmarks in 57 Atari Games', 'https://towardsdatascience.com/deepmind-unveils-agent57-the-first-ai-agents-that-outperforms-human-benchmarks-in-57-atari-games-35db4282dab3'),
               ])
ngu.connect(agent57)

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
root_policy_gradient.connect(reinforce, style=ROOT_EDGE)

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
root_policy_gradient.connect(dpg, style=ROOT_EDGE)

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
root_policy_gradient.connect(trpo, style=ROOT_EDGE)

gae = Node('GAE',
           'Generalized Advantage Estimation',
           policy_gradient,
           flags=[Flag.ONP, Flag.CS, Flag.CA],
           authors='John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel',
           year=2015,
           url='https://arxiv.org/abs/1506.02438',
           links=[('Generalized Advantage Estimator Explained', 'https://notanymike.github.io/GAE/'),
                  ('Notes on the Generalized Advantage Estimation Paper', 'https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/')]
           )
root_policy_gradient.connect(gae, style=ROOT_EDGE)
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
root_policy_gradient.connect(a3c, style=ROOT_EDGE)
a3c.connect(rainbow, style=WEAK_LINK, constraing='false')

ddpg_her = Node('DDPG+HER',
           'Hindsight Experience Replay (HER)',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.DP, Flag.RB],
           authors='Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, Wojciech Zaremba',
           year=2017,
           url='https://arxiv.org/abs/1707.01495',
           links=['https://becominghuman.ai/learning-from-mistakes-with-hindsight-experience-replay-547fce2b3305'])
ddpg.connect(ddpg_her, style=WEAK_LINK)
dqn_her.connect(ddpg_her, style=WEAK_LINK, label='HER', constraint='false', arrowhead='none')

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
dqn.connect(acer, style=WEAK_LINK, label='replay buffer')
# a2c.connect(acer, style=WEAK_LINK, label='multiple workers')
a2c.connect(acer, style=ORDER_EDGE)
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
root_policy_gradient.connect(acktr, style=ROOT_EDGE)
a2c.connect(acktr, style=ORDER_EDGE)  # just to maintain relative timeline order

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
root_policy_gradient.connect(svpg, style=ROOT_EDGE)
a2c.connect(svpg, style=ORDER_EDGE)  # just to maintain relative timeline order

reactor = Node('Reactor',
               'From the abstract: In this work we present a new agent architecture, called Reactor, which combines multiple algorithmic and architectural contributions to produce an agent with higher sample-efficiency than Prioritized Dueling DQN (Wang et al., 2016) and Categorical DQN (Bellemare et al., 2017), while giving better run-time performance than A3C (Mnih et al., 2016). Our first contribution is a new policy evaluation algorithm called Distributional Retrace, which brings multi-step off-policy updates to the distributional reinforcement learning setting. The same approach can be used to convert several classes of multi-step policy evaluation algorithms designed for expected value evaluation into distributional ones. Next, we introduce the β-leave-one-out policy gradient algorithm which improves the trade-off between variance and bias by using action values as a baseline. Our final algorithmic contribution is a new prioritized replay algorithm for sequences, which exploits the temporal locality of neighboring observations for more efficient replay prioritization. Using the Atari 2600 benchmarks, we show that each of these innovations contribute to both the sample efficiency and final agent performance. Finally, we demonstrate that Reactor reaches state-of-the-art performance after 200 million frames and less than a day of training.',
           policy_gradient,
           flags=[Flag.OFP, Flag.RNN, Flag.RB, Flag.DI],
           authors='Audrunas Gruslys, Will Dabney, Mohammad Gheshlaghi Azar, Bilal Piot, Marc Bellemare, Remi Munos',
           year=2017,
           url='https://arxiv.org/abs/1704.04651',
           links=[]
           )
root_policy_gradient.connect(reactor, style=ROOT_EDGE)


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

apex_ddpg = Node('APE-X DDPG',
           'DDPG with Distributed Prioritized Experience Replay',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.DA, Flag.RB],
           authors='Dan Horgan, John Quan, David Budden, Gabriel Barth-Maron, Matteo Hessel, Hado van Hasselt, David Silver',
           year=2018,
           url='https://arxiv.org/abs/1803.00933',
           links=[('Understanding and Implementing Distributed Prioritized Experience Replay (Horgan et al., 2018)', 'https://towardsdatascience.com/understanding-and-implementing-distributed-prioritized-experience-replay-horgan-et-al-2018-d2c1640e0520')])
ddpg.connect(apex_ddpg)
apex_dqn.connect(apex_ddpg, label='APE-X', style=WEAK_LINK, constraint='false', arrowhead='none')

sac = Node('SAC',
           'Soft Actor Critic (SAC) is an algorithm that optimizes a stochastic policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches.',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.CA, Flag.SP],
           authors='Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine',
           year=2018,
           url='https://arxiv.org/abs/1801.01290',
           links=[('Spinning Up SAC page', 'https://spinningup.openai.com/en/latest/algorithms/sac.html'),
                  ('(GitHub) SAC code by its author', 'https://github.com/haarnoja/sac')])
root_policy_gradient.connect(sac, style=ROOT_EDGE)
ppo.connect(sac, style=ORDER_EDGE)  # just to maintain relative timeline order

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

mpo = Node('MPO',
           'Maximum a Posteriori Policy Optimization (MPO) is an RL method that combines the sample efficiency of off-policy methods with the scalability and  hyperparameter robustness of  on-policy methods.  It is an EM style method, which alternates an E-step that re-weights state-action samples with an M step that updates a deep neural network with supervised training.  MPO achieves state of the art results on many continuous control tasks while using an order of magnitude fewer samples when compared with PPO',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.CA, Flag.DA],
           authors='Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, Martin Riedmiller',
           year=2018,
           url='https://arxiv.org/abs/1806.06920',
           links=[])
root_policy_gradient.connect(mpo, style=ROOT_EDGE)


impala = Node('IMPALA',
           'Importance Weighted Actor-Learner Architecture (IMPALA)',
           policy_gradient,
           flags=[Flag.OFP, Flag.CS, Flag.CA],
           authors='Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu',
           year=2018,
           url='https://arxiv.org/abs/1802.01561',
           links=[('Policy Gradient Algorithms', 'https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html')])
root_policy_gradient.connect(impala, style=ROOT_EDGE)
a2c.connect(impala, style=ORDER_EDGE)  # just to maintain relative timeline order

#
# MODEL BASED
#
dyna_q = Node('Dyna-Q',
            'Dyna-Q uses the experience drawn from real interaction with the environment to improve the value function/policy (called direct RL, using Q-learning) and the model of the environment (called model learning). The model is then used to create experiences (called planning) to improve the value function/policy.',
             model_based,
             flags=[Flag.OFP, Flag.MB],
             authors='Richard S. Sutton, Andrew G. Barto',
             year=1990,
             links=[('(book) Reinforcement Learning: An Introduction - 2nd Edition - Richard S. Sutton and Andrew G. Barto - Section 8.2', 'http://incompleteideas.net/book/the-book.html'),
                    ])
root_model_based.connect(dyna_q, style=ROOT_EDGE)

prio_sweep = Node('Prioritized Sweeping',
            'Prioritized Sweeping/Queue-Dyna is similar to Dyna, and it improves Dyna by updating value based on priority rather than randomly. Values are also associated with state rather than state-action.',
             model_based,
             flags=[Flag.MB],
             authors='Moore, Atkeson, Peng, Williams',
             year=1993,
             links=[('(book) Reinforcement Learning: An Introduction - 2nd Edition - Richard S. Sutton and Andrew G. Barto - Section 8.4', 'http://incompleteideas.net/book/the-book.html')])
dyna_q.connect(prio_sweep, style=ROOT_EDGE)

mcts = Node('MCTS',
            'Monte Carlo Tree Search (MCTS) selects the next action by performing rollout algorithm, which estimates action values for a given policy by averaging the returns of many simulated trajectories that start with each possible action and then follow the given policy. Unlike Monte Carlo control, the goal of a rollout algorithm is not to estimate a complete optimal action-value function, q-star, or a complete action-value function,q-pi, for a given policy pi. Instead, they produce Monte Carlo estimates of action values only for each current state, and once an action is selected, this estimation will be discarded and fresh calculation will be performed on the next state. MCTS enchances this rollout algorithm by the addition of a means for accumulating value estimates obtained from the Monte Carlo simulations in order to successively direct simulations toward more highly-rewarding trajectories.',
             model_based,
             flags=[Flag.MB],
             authors='Rémi Coulom, L. Kocsis, Cs. Szepesvári',
             year=2006,
             links=[('(book) Reinforcement Learning: An Introduction - 2nd Edition - Richard S. Sutton and Andrew G. Barto - Section 8.11', 'http://incompleteideas.net/book/the-book.html'),
                    ('(Wikipedia) MCTS', 'https://en.wikipedia.org/wiki/Monte_Carlo_tree_search')])
root_model_based.connect(mcts, style=ROOT_EDGE)

pilco = Node('PILCO',
            '(from the abstract) In this paper, we introduce PILCO, a practical, data-efficient model-based policy search method. PILCO reduces model bias, one of the key problems of model-based reinforcement learning, in a principled way.  By learning  a  probabilistic  dynamics  model  and  explicitly incorporating model uncertainty into long-term  planning,  PILCO can  cope  with very little data and facilitates learning froms cratch in only a few trials.  Policy evaluationis  performed  in  closed  form  using  state-of-the-art approximate inference.  Furthermore, policy  gradients  are  computed  analytically for policy improvement.  We report unprecedented learning efficiency on challenging and high-dimensional control tasks.',
             model_based,
             flags=[],
             authors='Marc Peter Deisenroth, Carl Edward Rasmussen',
             year=2011,
             url='https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Deisenroth_ICML_2011.pdf',
             links=[('PILCO website', 'http://mlg.eng.cam.ac.uk/pilco/'),
                    ])
root_model_based.connect(pilco, style=ROOT_EDGE)

i2a = Node('I2A',
            '(from the abstract) We introduce Imagination-Augmented Agents (I2As), a novel architecture for deep reinforcement learning combining model-free and model-based aspects. In contrast to most existing model-based reinforcement learning and planning methods, which prescribe how a model should be used to arrive at a policy, I2As learn to interpret predictions from a learned environment model to construct implicit plans in arbitrary ways, by using the predictions as additional context in deep policy networks. I2As show improved data efficiency, performance, and robustness to model misspecification compared to several baselines.',
             model_based,
             flags=[Flag.ML],
             authors='Théophane Weber, Sébastien Racanière, David P. Reichert, Lars Buesing, Arthur Guez, Danilo Jimenez Rezende, Adria Puigdomènech Badia, Oriol Vinyals, Nicolas Heess, Yujia Li, Razvan Pascanu, Peter Battaglia, Demis Hassabis, David Silver, Daan Wierstra',
             year=2017,
             url='https://arxiv.org/abs/1707.06203',
             links=[])
root_model_based.connect(i2a, style=ROOT_EDGE)

mbmf = Node('MBMF',
            '(from the abstract) Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning. We demonstrate that medium-sized neural network models can in fact be combined with model predictive control (MPC) to achieve excellent sample complexity in a model-based reinforcement learning algorithm, producing stable and plausible gaits to accomplish various complex locomotion tasks. We also propose using deep neural network dynamics models to initialize a model-free learner, in order to combine the sample efficiency of model-based approaches with the high task-specific performance of model-free methods. We empirically demonstrate on MuJoCo locomotion tasks that our pure model-based approach trained on just random action data can follow arbitrary trajectories with excellent sample efficiency, and that our hybrid algorithm can accelerate model-free learning on high-speed benchmark tasks, achieving sample efficiency gains of 3-5x on swimmer, cheetah, hopper, and ant agents.',
             model_based,
             flags=[Flag.ML],
             authors='Anusha Nagabandi, Gregory Kahn, Ronald S. Fearing, Sergey Levine',
             year=2017,
             url='https://arxiv.org/abs/1708.02596',
             links=[('Algorithm\'s site', 'https://sites.google.com/view/mbmf'),
                    ('(GitHub) Code', 'https://github.com/nagaban2/nn_dynamics'), ])
root_model_based.connect(mbmf, style=ROOT_EDGE)

exit_algo = Node('Exit',
             'Expert Iteration (ExIt) is a novel reinforcement learning algorithm which decomposes the problem into separate planning and generalisation tasks. Planning new policies is performed by tree search, while a deep neural network generalises those plans. Subsequently, tree search is improved by using the neural network policy to guide search, increasing the strength of new plans. In contrast, standard deep Reinforcement Learning algorithms rely on a neural network not only to generalise plans, but to discover them too. We show that ExIt outperforms REINFORCE for training a neural network to play the board game Hex, and our final tree search agent, trained tabula rasa, defeats MoHex 1.0, the most recent Olympiad Champion player to be publicly released. (from the abstract)',
             model_based,
             flags=[Flag.MG],
             authors='Thomas Anthony, Zheng Tian, David Barber',
             year=2017,
             url='https://arxiv.org/abs/1705.08439',
             links=[])
root_model_based.connect(exit_algo, style=ROOT_EDGE)

alpha_zero = Node('AlphaZero',
             'AlphaZero generalises tabula rasa reinforcement learning from games of self-play approach. Starting from random play, and given no domain knowledge except the game rules, AlphaZero achieved within 24 hours a superhuman level of play in the games of chess and shogi (Japanese chess) as well as Go, and convincingly defeated a world-champion program in each case. (from the abstract)',
             model_based,
             flags=[Flag.MG],
             authors='David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis',
             year=2017,
             url='https://arxiv.org/abs/1712.01815',
             links=[])
root_model_based.connect(alpha_zero, style=ROOT_EDGE)

mve = Node('MVE',
            '(from the abstract) Recent model-free reinforcement learning algorithms have proposed incorporating learned dynamics models as a source of additional data with the intention of reducing sample complexity. Such methods hold the promise of incorporating imagined data coupled with a notion of model uncertainty to accelerate the learning of continuous control tasks. Unfortunately, they rely on heuristics that limit usage of the dynamics model. We present model-based value expansion, which controls for uncertainty in the model by only allowing imagination to fixed depth. By enabling wider use of learned dynamics models within a model-free reinforcement learning algorithm, we improve value estimation, which, in turn, reduces the sample complexity of learning.',
             model_based,
             flags=[Flag.ML],
             authors='Vladimir Feinberg, Alvin Wan, Ion Stoica, Michael I. Jordan, Joseph E. Gonzalez, Sergey Levine',
             year=2018,
             url='https://arxiv.org/abs/1803.00101',
             links=[])
root_model_based.connect(mve, style=ROOT_EDGE)

steve = Node('STEVE',
            '(from the abstract) Integrating model-free and model-based approaches in reinforcement learning has the potential to achieve the high performance of model-free algorithms with low sample complexity. However, this is difficult because an imperfect dynamics model can degrade the performance of the learning algorithm, and in sufficiently complex environments, the dynamics model will almost always be imperfect. As a result, a key challenge is to combine model-based approaches with model-free learning in such a way that errors in the model do not degrade performance. We propose stochastic ensemble value expansion (STEVE), a novel model-based technique that addresses this issue. By dynamically interpolating between model rollouts of various horizon lengths for each individual example, STEVE ensures that the model is only utilized when doing so does not introduce significant errors. Our approach outperforms model-free baselines on challenging continuous control benchmarks with an order-of-magnitude increase in sample efficiency, and in contrast to previous model-based approaches, performance does not degrade in complex environments.',
             model_based,
             flags=[Flag.ML],
             authors='Jacob Buckman, Danijar Hafner, George Tucker, Eugene Brevdo, Honglak Lee',
             year=2018,
             url='https://arxiv.org/abs/1807.01675',
             links=[])
root_model_based.connect(steve, style=ROOT_EDGE)

me_trpo = Node('ME-TRPO',
            '(from the abstract) Model-free reinforcement learning (RL) methods are succeeding in a growing number of tasks, aided by recent advances in deep learning. However, they tend to suffer from high sample complexity, which hinders their use in real-world domains. Alternatively, model-based reinforcement learning promises to reduce sample complexity, but tends to require careful tuning and to date have succeeded mainly in restrictive domains where simple models are sufficient for learning. In this paper, we analyze the behavior of vanilla model-based reinforcement learning methods when deep neural networks are used to learn both the model and the policy, and show that the learned policy tends to exploit regions where insufficient data is available for the model to be learned, causing instability in training. To overcome this issue, we propose to use an ensemble of models to maintain the model uncertainty and regularize the learning process. We further show that the use of likelihood ratio derivatives yields much more stable learning than backpropagation through time. Altogether, our approach Model-Ensemble Trust-Region Policy Optimization (ME-TRPO) significantly reduces the sample complexity compared to model-free deep RL methods on challenging continuous control benchmark tasks.',
             model_based,
             flags=[Flag.ML],
             authors='Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, Pieter Abbeel',
             year=2018,
             url='https://arxiv.org/abs/1802.10592',
             links=[])
root_model_based.connect(me_trpo, style=ROOT_EDGE)

mb_mpo = Node('MB-MPO',
            '(from the abstract) Model-based reinforcement learning approaches carry the promise of being data efficient. However, due to challenges in learning dynamics models that sufficiently match the real-world dynamics, they struggle to achieve the same asymptotic performance as model-free methods. We propose Model-Based Meta-Policy-Optimization (MB-MPO), an approach that foregoes the strong reliance on accurate learned dynamics models. Using an ensemble of learned dynamic models, MB-MPO meta-learns a policy that can quickly adapt to any model in the ensemble with one policy gradient step. This steers the meta-policy towards internalizing consistent dynamics predictions among the ensemble while shifting the burden of behaving optimally w.r.t. the model discrepancies towards the adaptation step. Our experiments show that MB-MPO is more robust to model imperfections than previous model-based approaches. Finally, we demonstrate that our approach is able to match the asymptotic performance of model-free methods while requiring significantly less experience. ',
             model_based,
             flags=[Flag.ML],
             authors='Ignasi Clavera, Jonas Rothfuss, John Schulman, Yasuhiro Fujita, Tamim Asfour, Pieter Abbeel',
             year=2018,
             url='https://arxiv.org/abs/1809.05214',
             links=[])
root_model_based.connect(mb_mpo, style=ROOT_EDGE)

world_models = Node('World Models',
            '(from the abstract) A generative recurrent neural network is quickly trained in an unsupervised manner to model popular reinforcement learning environments through compressed spatio-temporal representations. The world model\'s extracted features are fed into compact and simple policies trained by evolution, achieving state of the art results in various environments. We also train our agent entirely inside of an environment generated by its own internal world model, and transfer this policy back into the actual environment.',
             model_based,
             flags=[Flag.ML],
             authors='David Ha, Jürgen Schmidhuber',
             year=2018,
             url='https://arxiv.org/abs/1809.01999',
             links=[('Interactive version of the paper', 'https://worldmodels.github.io/'),
                    ('The experiment code', 'https://blog.otoro.net/2018/06/09/world-models-experiments/')])
root_model_based.connect(world_models, style=ROOT_EDGE)

pets = Node('PETS',
            '(from the abstract) Model-based reinforcement learning (RL) algorithms can attain excellent sample efficiency, but often lag behind the best model-free algorithms in terms of asymptotic performance. This is especially true with high-capacity parametric function approximators, such as deep networks. In this paper, we study how to bridge this gap, by employing uncertainty-aware dynamics models. We propose a new algorithm called probabilistic ensembles with trajectory sampling (PETS) that combines uncertainty-aware deep network dynamics models with sampling-based uncertainty propagation. Our comparison to state-of-the-art model-based and model-free deep RL algorithms shows that our approach matches the asymptotic performance of model-free algorithms on several challenging benchmark tasks, while requiring significantly fewer samples (e.g., 8 and 125 times fewer samples than Soft Actor Critic and Proximal Policy Optimization respectively on the half-cheetah task).',
             model_based,
             flags=[],
             authors='Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine',
             year=2018,
             url='https://arxiv.org/abs/1805.12114',
             links=[])
root_model_based.connect(pets, style=ROOT_EDGE)

planet = Node('PlaNet',
            '(from the abstract) We propose the Deep Planning Network (PlaNet), a purely model-based agent that learns the environment dynamics from images and chooses actions through fast online planning in latent space. To achieve high performance, the dynamics model must accurately predict the rewards ahead for multiple time steps. We approach this using a latent dynamics model with both deterministic and stochastic transition components. Moreover, we propose a multi-step variational inference objective that we name latent overshooting. Using only pixel observations, our agent solves continuous control tasks with contact dynamics, partial observability, and sparse rewards, which exceed the difficulty of tasks that were previously solved by planning with learned models. PlaNet uses substantially fewer episodes and reaches final performance close to and sometimes higher than strong model-free algorithms.',
             model_based,
             flags=[],
             authors='Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson',
             year=2018,
             url='https://arxiv.org/abs/1811.04551',
             links=[])
root_model_based.connect(planet, style=ROOT_EDGE)

simple = Node('SimPLe',
             'Simulated Policy Learning (SimPLe) is a complete model-based deep RL algorithm based on video prediction models and present a comparison of several model architectures, including a novel architecture that yields the best results in our setting. Our experiments evaluate SimPLe on a range of Atari games in low data regime of 100k interactions between the agent and the environment, which corresponds to two hours of real-time play. In most games SimPLe outperforms state-of-the-art model-free algorithms, in some games by over an order of magnitude. (from the abstract)',
             model_based,
             flags=[Flag.ML],
             authors='Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, Afroz Mohiuddin, Ryan Sepassi, George Tucker, Henryk Michalewski',
             year=2019,
             url='https://arxiv.org/abs/1903.00374')
root_model_based.connect(simple, style=ROOT_EDGE)

muzero = Node('MuZero',
             '(from the abstract) Constructing agents with planning capabilities has long been one of the main challenges in the pursuit of artificial intelligence. Tree-based planning methods have enjoyed huge success in challenging domains, such as chess and Go, where a perfect simulator is available. However, in real-world problems the dynamics governing the environment are often complex and unknown. In this work we present the MuZero algorithm which, by combining a tree-based search with a learned model, achieves superhuman performance in a range of challenging and visually complex domains, without any knowledge of their underlying dynamics. MuZero learns a model that, when applied iteratively, predicts the quantities most directly relevant to planning: the reward, the action-selection policy, and the value function. When evaluated on 57 different Atari games - the canonical video game environment for testing AI techniques, in which model-based planning approaches have historically struggled - our new algorithm achieved a new state of the art. When evaluated on Go, chess and shogi, without any knowledge of the game rules, MuZero matched the superhuman performance of the AlphaZero algorithm that was supplied with the game rules. ',
             model_based,
             flags=[Flag.ML],
             authors='Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, David Silver',
             year=2019,
             url='https://arxiv.org/abs/1911.08265')
root_model_based.connect(muzero, style=ROOT_EDGE)

#
# META-RL
#
dmrl = Node('DMRL',
            'Deep Meta RL. (from the abstract) In recent years deep reinforcement learning (RL) systems have attained superhuman performance in a number of challenging task domains. However, a major limitation of such applications is their demand for massive amounts of training data. A critical present objective is thus to develop deep RL methods that can adapt rapidly to new tasks. In the present work we introduce a novel approach to this challenge, which we refer to as deep meta-reinforcement learning. Previous work has shown that recurrent networks can support meta-learning in a fully supervised context. We extend this approach to the RL setting. What emerges is a system that is trained using one RL algorithm, but whose recurrent dynamics implement a second, quite separate RL procedure. This second, learned RL algorithm can differ from the original one in arbitrary ways. Importantly, because it is learned, it is configured to exploit structure in the training domain. We unpack these points in a series of seven proof-of-concept experiments, each of which examines a key aspect of deep meta-RL. We consider prospects for extending and scaling up the approach, and also point out some potentially important implications for neuroscience. ',
             meta_rl,
             flags=[],
             authors='Jane X Wang, Zeb Kurth-Nelson, Dhruva Tirumala, Hubert Soyer, Joel Z Leibo, Remi Munos, Charles Blundell, Dharshan Kumaran, Matt Botvinick',
             year=2016,
             url='https://arxiv.org/abs/1611.05763',
             links=[])
root_meta_rl.connect(dmrl, style=ROOT_EDGE)

rl2 = Node('RL^2',
            '(from the abstract) Deep reinforcement learning (deep RL) has been successful in learning sophisticated behaviors automatically; however, the learning process requires a huge number of trials. In contrast, animals can learn new tasks in just a few trials, benefiting from their prior knowledge about the world. This paper seeks to bridge this gap. Rather than designing a "fast" reinforcement learning algorithm, we propose to represent it as a recurrent neural network (RNN) and learn it from data. In our proposed method, RL<sup>2</sup>, the algorithm is encoded in the weights of the RNN, which are learned slowly through a general-purpose ("slow") RL algorithm. The RNN receives all information a typical RL algorithm would receive, including observations, actions, rewards, and termination flags; and it retains its state across episodes in a given Markov Decision Process (MDP). The activations of the RNN store the state of the "fast" RL algorithm on the current (previously unseen) MDP. We evaluate RL<sup>2</sup> experimentally on both small-scale and large-scale problems. On the small-scale side, we train it to solve randomly generated multi-arm bandit problems and finite MDPs. After RL<sup>2</sup> is trained, its performance on new MDPs is close to human-designed algorithms with optimality guarantees. On the large-scale side, we test RL<sup>2</sup> on a vision-based navigation task and show that it scales up to high-dimensional problems.',
             meta_rl,
             flags=[],
             authors='Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, Pieter Abbeel',
             year=2016,
             url='https://arxiv.org/abs/1611.02779',
             links=[])
root_meta_rl.connect(rl2, style=ROOT_EDGE)

maml = Node('MAML',
            '(from the abstract) We propose an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning. The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task. In effect, our method trains the model to be easy to fine-tune. We demonstrate that this approach leads to state-of-the-art performance on two few-shot image classification benchmarks, produces good results on few-shot regression, and accelerates fine-tuning for policy gradient reinforcement learning with neural network policies.',
             meta_rl,
             flags=[],
             authors='Chelsea Finn, Pieter Abbeel, Sergey Levine',
             year=2017,
             url='https://arxiv.org/abs/1703.03400',
             links=[('Learning to Learn', 'https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/'),
                    ('(GitHub) Code for MAML', 'https://github.com/cbfinn/maml')],
             )
root_meta_rl.connect(maml, style=ROOT_EDGE)

snail = Node('SNAIL',
            '(from the abstract) Deep neural networks excel in regimes with large amounts of data, but tend to struggle when data is scarce or when they need to adapt quickly to changes in the task. In response, recent work in meta-learning proposes training a meta-learner on a distribution of similar tasks, in the hopes of generalization to novel but related tasks by learning a high-level strategy that captures the essence of the problem it is asked to solve. However, many recent meta-learning approaches are extensively hand-designed, either using architectures specialized to a particular application, or hard-coding algorithmic components that constrain how the meta-learner solves the task. We propose a class of simple and generic meta-learner architectures that use a novel combination of temporal convolutions and soft attention; the former to aggregate information from past experience and the latter to pinpoint specific pieces of information. In the most extensive set of meta-learning experiments to date, we evaluate the resulting Simple Neural AttentIve Learner (or SNAIL) on several heavily-benchmarked tasks. On all tasks, in both supervised and reinforcement learning, SNAIL attains state-of-the-art performance by significant margins.',
             meta_rl,
             flags=[],
             authors='Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, Pieter Abbeel',
             year=2017,
             url='https://arxiv.org/abs/1707.03141',
             links=[('A Simple Neural Attentive Meta-Learner — SNAIL', 'https://medium.com/towards-artificial-intelligence/a-simple-neural-attentive-meta-learner-snail-1e6b1d487623')],
             )
root_meta_rl.connect(snail, style=ROOT_EDGE)

pro_mp = Node('ProMP',
            'ProMP: Proximal Meta-Policy Search (from the abstract) Credit assignment in Meta-reinforcement learning (Meta-RL) is still poorly understood. Existing methods either neglect credit assignment to pre-adaptation behavior or implement it naively. This leads to poor sample-efficiency during meta-training as well as ineffective task identification strategies. This paper provides a theoretical analysis of credit assignment in gradient-based Meta-RL. Building on the gained insights we develop a novel meta-learning algorithm that overcomes both the issue of poor credit assignment and previous difficulties in estimating meta-policy gradients. By controlling the statistical distance of both pre-adaptation and adapted policies during meta-policy search, the proposed algorithm endows efficient and stable meta-learning. Our approach leads to superior pre-adaptation policy behavior and consistently outperforms previous Meta-RL algorithms in sample-efficiency, wall-clock time, and asymptotic performance. ',
             meta_rl,
             flags=[],
             authors='Jonas Rothfuss, Dennis Lee, Ignasi Clavera, Tamim Asfour, Pieter Abbeel',
             year=2018,
             url='https://arxiv.org/abs/1810.06784',
             links=[],
             )
root_meta_rl.connect(pro_mp, style=ROOT_EDGE)


def generate_graph(output, format):
    from graphviz import Digraph, Source

    graph = Digraph()
    graph.attr(compound='true')
    graph.attr(rankdir=RANKDIR)
    graph.attr(newrank='true')  # need this to enable rank=same across clusters

    # Timeline
    ranks = OrderedDict()
    nodes = rl.collect_nodes()
    nodes = sorted(nodes, key=lambda n: n.graph_rank)
    for node in nodes:
        lst = ranks.get(node.graph_rank, [])
        lst.append(node)
        ranks[node.graph_rank] = lst

    START_TIMELINE = "1950s"

    # The global timeline graph
    with graph.subgraph(name='clusterTimeline') as timeline_graph:
        timeline_graph.attr(shape='box')
        timeline_graph.attr(style='bold,filled')
        timeline_graph.attr(fillcolor=TIMELINE_FILL_COLOR)
        timeline_graph.attr(color=TIMELINE_FILL_COLOR)
        timeline_graph.attr(margin='0')
        timeline_graph.attr(pad='0')
        years = [START_TIMELINE] + [k for k in ranks.keys() if k and k >= '1900']
        for year in years:
            if RANKDIR=='LR':
                attrs = {'height': '0.2'}
            else:
                attrs = {}
            timeline_graph.node(f'{year}', fontcolor=TIMELINE_COLOR, shape='plaintext',
                                fontname=TIMELINE_FONT_NAME, fontsize=str(TIMELINE_FONT_SIZE),
                                margin="0", pad="0", **attrs,
                                group=f'timeline')  # use same group to make straight nodes
        for iy in range(len(years) - 1):
            timeline_graph.edge(str(years[iy]), str(years[iy + 1]), color=TIMELINE_COLOR)

    rl.export_graph(graph, graph)

    # Create cluster to align nodes in the same year
    for rank, members in ranks.items():
        if not rank or rank < '1900':
            continue
        with graph.subgraph() as rank_cluster:
            rank_cluster.attr(rank='same')
            rank_cluster.node(f'{rank}')
            for node in members:
                rank_cluster.node(node.graph_name)

    # Align value gradient and policy gradient
    with graph.subgraph() as rank_cluster:
        rank_cluster.attr(rank='same')
        rank_cluster.node(START_TIMELINE)
        rank_cluster.node(rl.graph_name)

    # Align value gradient and policy gradient
    with graph.subgraph() as rank_cluster:
        rank_cluster.attr(rank='same')
        rank_cluster.node(root_model_free.graph_name)
        rank_cluster.node(root_model_based.graph_name)
        #rank_cluster.node(root_meta_rl.graph_name)

    with graph.subgraph() as rank_cluster:
        rank_cluster.attr(rank='same')
        rank_cluster.node(root_value_gradient.graph_name)
        rank_cluster.node(root_policy_gradient.graph_name)


    graph.render(output, format=format)


def generate_md():
    nodes = rl.collect_nodes()

    md = """# RL Taxonomy

This is a loose taxonomy of reinforcement learning algorithms. I'm by no means expert in this area, I'm making this as part of my learning process. Note that there are a lot more algorithms than listed here, and often I don't even know how to categorize them. In any case, please PR to correct things or suggest new stuff.

Note that this file is generated by `taxonomy.py`.
"""
    md += '#### Table of Contents:<HR>\n\n'
    md += "[Taxonomy](#taxonomy)<BR>\n"
    for node in nodes:
        if not node.output_md:
            continue
        parents = node.get_parents()
        if len(parents):
            md += '  ' * (len(parents) - 1)
        if parents:
            md += '- '
        md += f'[{node.md_title}](#{node.name})\n'
    md += '\n'

    md += f"""## <A name="taxonomy"></a>Taxonomy

Below is the taxonomy of reinforcement learning algorithms. Solid line indicates some progression from one idea to another. Dashed line indicates a loose connection. On the {"bottom" if RANKDIR=="LR" else "left"} you can see the timeline of the publication year of the algorithms. 

It's recommended to open the .SVG file in a new window, as hovering the mouse over the algorithm will show tooltip containing the description of the algorithm and clicking the node will open the link to its description.

![RL Taxonomy](rl-taxonomy.gv.svg "RL Taxonomy")\n\n"""

    md += rl.export_md()

    md += """<HR>
    
Sources:
- [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
- [Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- [Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)

"""

    md += '\n(This document is autogenerated)\n'
    return md


if __name__ == '__main__':
    with open('README.md', 'w', encoding="utf-8") as f:
        f.write(generate_md())

    generate_graph('rl-taxonomy.gv', 'svg')
