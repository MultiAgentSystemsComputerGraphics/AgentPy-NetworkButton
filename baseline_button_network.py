# baseline_button_network.py
# pip install agentpy==0.1.6.dev0 networkx matplotlib numpy pandas

import agentpy as ap
import networkx as nx
import numpy as np

class BaselineButtonModel(ap.Model):
    """Baseline Button Network:
    - n nodes
    - Each step adds ~ n*speed random edges
    - Records giant component vs. threads/buttons ratio
    """

    def setup(self):
        # Do NOT call self.set_seed(...). Experiment injects seeds for each run.
        self.net = ap.Network(self)
        self.agents = ap.AgentList(self, self.p.n)
        self.net.add_agents(self.agents)
        self.threads = 0

    def step(self):
        m = int(self.p.n * self.p.speed)
        for _ in range(m):
            a, b = self.random.sample(self.agents, 2)   # use AgentPy RNG
            self.net.graph.add_edge(a, b)
            self.threads += 1

        comps = nx.connected_components(self.net.graph)
        largest = max((len(c) for c in comps), default=1)
        frac = largest / self.p.n
        self.record('giant_frac', frac)
        self.record('threads_to_button', self.threads / self.p.n)
        self.record('mean_degree', np.mean([d for _, d in self.net.graph.degree()]))
        self.record('clustering', nx.transitivity(self.net.graph))

    def end(self):
        gf = np.array(self.log['giant_frac'])
        tb = np.array(self.log['threads_to_button'])
        above = np.where(gf >= 0.5)[0]
        self.report('threshold_t_over_b',
                    float(tb[above[0]]) if len(above) else float('nan'))
