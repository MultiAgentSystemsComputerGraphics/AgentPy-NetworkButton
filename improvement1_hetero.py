import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"networkx\.algorithms\.assortativity")

import agentpy as ap
import networkx as nx
import numpy as np

class ButtonHeteroHomophily(ap.Model):
    def setup(self):
        self.net = ap.Network(self)
        self.agents = ap.AgentList(self, self.p.n)
        self.net.add_agents(self.agents)

        # NumPy RNG seeded from AgentPy RNG
        np_seed = self.random.randint(0, 2**32 - 1)
        self.nprng = np.random.default_rng(np_seed)

        groups = int(self.p.groups)
        group_list = [self.random.randrange(groups) for _ in range(self.p.n)]
        activity_list = self.nprng.lognormal(self.p.activity_mu,
                                             self.p.activity_sigma,
                                             self.p.n)

        for a, g, act in zip(self.agents, group_list, activity_list):
            a.group = int(g)
            a.activity = float(act)

        # write node attribute using actual node keys
        mapping = {node: a.group for node, a in zip(self.net.graph.nodes, self.agents)}
        nx.set_node_attributes(self.net.graph, mapping, "group")

        self.homophily = float(self.p.homophily)
        self.threads = 0

    # ----- weighted helpers using NumPy RNG -----
    def _weighted_choice(self, seq, p):
        idx = self.nprng.choice(len(seq), p=p)
        return seq[idx]

    def _pick_initiator(self):
        w = np.array([a.activity for a in self.agents], dtype=float)
        s = w.sum()
        if s <= 0:
            return self.random.choice(self.agents)   # unweighted fallback
        w /= s
        return self._weighted_choice(self.agents, w)

    def _pick_partner(self, initiator):
        ig = initiator.group
        scores = np.array([
            self.homophily if b.group == ig else (1.0 - self.homophily)
            for b in self.agents
        ], dtype=float)
        scores[self.agents.index(initiator)] = 0.0
        s = scores.sum()
        if s <= 0:
            return None
        scores /= s
        return self._weighted_choice(self.agents, scores)

    def step(self):
        attempts = int(self.p.n * self.p.speed)
        for _ in range(attempts):
            i = self._pick_initiator()
            j = self._pick_partner(i)
            if j is None:
                continue
            self.net.graph.add_edge(i, j)
            self.threads += 1

        comps = nx.connected_components(self.net.graph)
        largest = max((len(c) for c in comps), default=1) / self.p.n
        self.record('giant_frac', largest)
        self.record('threads_to_button', self.threads / self.p.n)

        degs = [d for _, d in self.net.graph.degree()]
        self.record('mean_degree', float(np.mean(degs)) if degs else 0.0)
        self.record('clustering', nx.transitivity(self.net.graph))
        try:
            assort = nx.attribute_assortativity_coefficient(self.net.graph, 'group')
        except Exception:
            assort = float('nan')
        self.record('group_assort', assort)

    def end(self):
        gf = np.array(self.log['giant_frac'])
        tb = np.array(self.log['threads_to_button'])
        above = np.where(gf >= 0.5)[0]
        self.report('threshold_t_over_b',
                    float(tb[above[0]]) if len(above) else float('nan'))
