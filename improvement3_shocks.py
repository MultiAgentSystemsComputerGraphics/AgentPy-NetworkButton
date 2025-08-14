import agentpy as ap
import networkx as nx
import numpy as np

class ButtonWithShocks(ap.Model):
    def setup(self):
        self.net = ap.Network(self)
        self.agents = ap.AgentList(self, self.p.n)
        self.net.add_agents(self.agents)
        self.threads = 0

        # Build multiplier schedule (length = steps)
        self.mult = np.ones(self.p.steps, dtype=float)
        if self.p.get('rate_schedule') is not None:
            rs = np.array(self.p.rate_schedule, dtype=float)
            assert len(rs) == self.p.steps
            self.mult = rs
        else:
            for s in list(self.p.get('shock_steps', [])):  # accept tuple/list
                start = int(s)
                end = min(self.p.steps, start + int(self.p.get('shock_duration', 1)))
                self.mult[start:end] *= float(self.p.get('shock_multiplier', 2.0))

    def step(self):
        # Guard against self.t == steps in this AgentPy build
        idx = min(self.t, len(self.mult) - 1)
        m = int(self.p.n * self.p.speed * self.mult[idx])

        for _ in range(m):
            a, b = self.random.sample(self.agents, 2)
            self.net.graph.add_edge(a, b)
            self.threads += 1

        comps = nx.connected_components(self.net.graph)
        largest = max((len(c) for c in comps), default=1) / self.p.n
        self.record('giant_frac', largest)
        self.record('threads_to_button', self.threads / self.p.n)

    def end(self):
        gf = np.array(self.log['giant_frac'])
        tb = np.array(self.log['threads_to_button'])
        above = np.where(gf >= 0.5)[0]
        self.report('threshold_t_over_b',
                    float(tb[above[0]]) if len(above) else float('nan'))
