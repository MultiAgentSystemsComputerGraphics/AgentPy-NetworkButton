import agentpy as ap
import networkx as nx
import numpy as np

class ButtonWithCapacity(ap.Model):
    def setup(self):
        self.net = ap.Network(self)
        self.agents = ap.AgentList(self, self.p.n)
        self.net.add_agents(self.agents)
        self.threads = 0
        self.rejections = 0

        # Map each agent -> actual NetworkX node key (bullet-proof against key type)
        self.agent_to_node = {a: node for node, a in zip(self.net.graph.nodes, self.agents)}

        # NumPy RNG seeded from AgentPy RNG (for reproducibility)
        np_seed = self.random.randint(0, 2**32 - 1)
        self.nprng = np.random.default_rng(np_seed)

        # Degree capacities
        if self.p.capacity_sigma > 0:
            caps = np.maximum(1, self.nprng.normal(
                self.p.capacity_mu, self.p.capacity_sigma, self.p.n).astype(int))
        else:
            caps = np.full(self.p.n, int(self.p.capacity_mu), dtype=int)

        # store per agent (avoid AgentList broadcast)
        for a, c in zip(self.agents, caps):
            a.capacity = int(c)

    def _deg(self, a):
        """Degree of an agent's node (works even if node keys aren't agents)."""
        node = self.agent_to_node[a]
        return self.net.graph.degree[node]

    def _can_link(self, a, b):
        return (self._deg(a) < a.capacity) and (self._deg(b) < b.capacity)

    def step(self):
        m = int(self.p.n * self.p.speed)
        for _ in range(m):
            a, b = self.random.sample(self.agents, 2)
            if self._can_link(a, b):
                na = self.agent_to_node[a]
                nb = self.agent_to_node[b]
                self.net.graph.add_edge(na, nb)   # always use node keys
                self.threads += 1
            else:
                self.rejections += 1

        # KPIs
        comps = nx.connected_components(self.net.graph)
        largest = max((len(c) for c in comps), default=1) / self.p.n
        self.record('giant_frac', largest)
        self.record('threads_to_button', self.threads / self.p.n)

        degs = [self.net.graph.degree[n] for n in self.net.graph.nodes]
        self.record('mean_degree', float(np.mean(degs)) if degs else 0.0)
        self.record('clustering', nx.transitivity(self.net.graph))

        # Congestion KPIs
        caps = np.array([a.capacity for a in self.agents], dtype=int)
        degs_vec = np.array([self._deg(a) for a in self.agents], dtype=int)
        total = self.rejections + self.threads
        self.record('rejection_rate', self.rejections / total if total else 0.0)
        self.record('saturation_frac', float(np.mean(degs_vec >= caps)))

    def end(self):
        gf = np.array(self.log['giant_frac'])
        tb = np.array(self.log['threads_to_button'])
        above = np.where(gf >= 0.5)[0]
        self.report('threshold_t_over_b',
                    float(tb[above[0]]) if len(above) else float('nan'))
