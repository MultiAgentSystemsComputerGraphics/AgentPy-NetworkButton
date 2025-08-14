import agentpy as ap
import networkx as nx
import numpy as np
import warnings

# Optional: silence occasional assortativity warnings when the partition is degenerate
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"networkx\.algorithms\.assortativity")

class RealisticButtonModel(ap.Model):
    """Button Network with toggles:
       - Heterogeneity + Homophily
       - Capacity constraints
       - Exogenous shocks/time-varying rate
    """

    def setup(self):
        self.net = ap.Network(self)
        self.agents = ap.AgentList(self, self.p.n)
        self.net.add_agents(self.agents)
        self.threads = 0
        self.rejections = 0

        # Map each agent -> actual NetworkX node key (robust regardless of node key type)
        self.agent_to_node = {a: node for node, a in zip(self.net.graph.nodes, self.agents)}

        # NumPy RNG seeded from AgentPy RNG (for reproducibility)
        np_seed = self.random.randint(0, 2**32 - 1)
        self.nprng = np.random.default_rng(np_seed)

        # --- Heterogeneity ---
        self.use_hetero = bool(self.p.get('use_heterogeneity', False))
        if self.use_hetero:
            groups = int(self.p.get('groups', 2))
            group_list = [self.random.randrange(groups) for _ in range(self.p.n)]
            activity_list = self.nprng.lognormal(
                float(self.p.get('activity_mu', -1.0)),
                float(self.p.get('activity_sigma', 0.75)),
                self.p.n
            )
            for a, g, act in zip(self.agents, group_list, activity_list):
                a.group = int(g)
                a.activity = float(act)

            # set node attribute 'group' via mapping of actual node keys
            nx.set_node_attributes(
                self.net.graph,
                {self.agent_to_node[a]: a.group for a in self.agents},
                "group",
            )
            self.homophily = float(self.p.get('homophily', 0.7))

        # --- Capacity ---
        self.use_capacity = bool(self.p.get('use_capacity', False))
        if self.use_capacity:
            mu = float(self.p.get('capacity_mu', 12))
            sig = float(self.p.get('capacity_sigma', 0))
            if sig > 0:
                caps = np.maximum(1, self.nprng.normal(mu, sig, self.p.n).astype(int))
            else:
                caps = np.full(self.p.n, int(mu), dtype=int)
            for a, c in zip(self.agents, caps):
                a.capacity = int(c)

        # --- Shocks / rate schedule ---
        self.use_shocks = bool(self.p.get('use_shocks', False))
        self.mult = np.ones(self.p.steps, dtype=float)
        if self.use_shocks:
            if self.p.get('rate_schedule') is not None:
                rs = np.array(self.p.rate_schedule, dtype=float)
                assert len(rs) == self.p.steps
                self.mult = rs
            else:
                for s in list(self.p.get('shock_steps', [])):  # accept tuple/list
                    start = int(s)
                    end = min(self.p.steps, start + int(self.p.get('shock_duration', 1)))
                    self.mult[start:end] *= float(self.p.get('shock_multiplier', 2.0))

    # ----- helpers -----
    def _weighted_choice(self, seq, p):
        idx = self.nprng.choice(len(seq), p=p)
        return seq[idx]

    def _pick_initiator(self):
        if not self.use_hetero:
            return self.random.choice(self.agents)
        w = np.array([a.activity for a in self.agents], dtype=float)
        s = w.sum()
        if s <= 0:
            return self.random.choice(self.agents)
        w /= s
        return self._weighted_choice(self.agents, w)

    def _pick_partner(self, initiator):
        if not self.use_hetero:
            pool = [a for a in self.agents if a is not initiator]
            return self.random.choice(pool)
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

    def _deg(self, a):
        """Degree via node key (never index graph with the agent object)."""
        return self.net.graph.degree[self.agent_to_node[a]]

    def _can_link(self, a, b):
        if not self.use_capacity:
            return True
        return (self._deg(a) < getattr(a, 'capacity', np.inf)) and (self._deg(b) < getattr(b, 'capacity', np.inf))

    def step(self):
        # Guard against self.t == steps in this AgentPy build
        idx = min(self.t, len(self.mult) - 1)
        mult = self.mult[idx] if self.use_shocks else 1.0
        m = int(self.p.n * self.p.speed * mult)

        for _ in range(m):
            if self.use_hetero:
                i = self._pick_initiator()
                j = self._pick_partner(i)
                if j is None:
                    continue
                a, b = i, j
            else:
                a, b = self.random.sample(self.agents, 2)

            if self._can_link(a, b):
                na = self.agent_to_node[a]
                nb = self.agent_to_node[b]
                self.net.graph.add_edge(na, nb)  # always use node keys
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

        if self.use_hetero:
            try:
                assort = nx.attribute_assortativity_coefficient(self.net.graph, 'group')
            except Exception:
                assort = float('nan')
            self.record('group_assort', assort)

        if self.use_capacity:
            caps = np.array([getattr(a, 'capacity', np.inf) for a in self.agents], dtype=float)
            degs_vec = np.array([self._deg(a) for a in self.agents], dtype=float)
            total = self.rejections + self.threads
            self.record('rejection_rate', self.rejections / total if total else 0.0)
            self.record('saturation_frac', float(np.mean(degs_vec >= caps)))

    def end(self):
        gf = np.array(self.log['giant_frac'])
        tb = np.array(self.log['threads_to_button'])
        above = np.where(gf >= 0.5)[0]
        self.report('threshold_t_over_b',
                    float(tb[above[0]]) if len(above) else float('nan'))
