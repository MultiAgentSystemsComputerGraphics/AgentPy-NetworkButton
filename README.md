# Button Network: Agent-Based Percolation Analysis

This project implements and analyzes various improvements to a baseline Button Network model using agent-based modeling to study percolation phenomena and network formation dynamics.

## Core Concepts

### What is a Button Network?

A **Button Network** is a mathematical model that simulates the formation of connections in a network through a threading process, analogous to sewing buttons together with threads. In this model:

- **Buttons** = Nodes/Agents in the network
- **Threads** = Edges/Connections between nodes  
- **Threading Process** = Dynamic addition of edges over time

The model starts with isolated nodes (buttons) and gradually adds connections (threads) until a giant connected component emerges.

### Agent-Based Modeling

**Agents** are autonomous entities that represent individual buttons in the network. Each agent can have:
- **Properties**: Group membership, activity levels, capacity constraints
- **Behaviors**: How they form connections with other agents
- **Decision rules**: Preference for connecting to similar agents (homophily)

### Percolation Theory

**Percolation** refers to the sudden emergence of a giant connected component when a critical threshold is reached. Key concepts:

- **Percolation Threshold**: The critical point where network connectivity rapidly increases
- **Giant Component**: The largest connected subgraph that contains a significant fraction of all nodes
- **Phase Transition**: The rapid change from disconnected fragments to a single large connected component

## Models and Improvements

### 1. Baseline Model (`baseline_button_network.py`)

The foundation model that establishes the basic percolation dynamics:

**Key Features:**
- `n` nodes (buttons) start completely disconnected
- Each time step adds `n × speed` random edges (threads)
- Tracks the fraction of nodes in the largest connected component
- Records the threads-to-buttons ratio at each step

**Metrics Tracked:**
- `giant_frac`: Fraction of nodes in the largest component
- `threads_to_button`: Ratio of total edges to total nodes
- `mean_degree`: Average number of connections per node
- `clustering`: Network clustering coefficient
- `threshold_t_over_b`: Threads/buttons ratio when giant component reaches 50%

### 2. Improvement 1: Heterogeneity & Homophily (`improvement1_hetero.py`)

Introduces **social realism** through agent diversity and preference-based connection formation.

**New Features:**
- **Heterogeneity**: Agents have different activity levels (log-normal distribution)
- **Group Membership**: Agents belong to different social groups
- **Homophily**: Agents prefer connecting to similar others (same group)
- **Activity-Based Selection**: More active agents initiate more connections

**Key Parameters:**
- `groups`: Number of social groups
- `activity_mu`, `activity_sigma`: Log-normal activity distribution parameters
- `homophily`: Probability of preferring same-group connections (0.0-1.0)

**Impact on Results:**
- Creates more clustered networks with higher assortativity
- May accelerate or delay percolation depending on group structure
- Introduces social stratification effects

### 3. Improvement 2: Capacity Constraints (`improvement2_capacity.py`)

Adds **realistic limitations** on how many connections each agent can maintain.

**New Features:**
- **Degree Capacity**: Each agent has a maximum number of allowed connections
- **Connection Rejection**: Attempts to connect capacity-saturated agents fail
- **Congestion Modeling**: Tracks rejection rates and saturation levels

**Key Parameters:**
- `capacity_mu`: Mean capacity per agent
- `capacity_sigma`: Standard deviation of capacity distribution

**New Metrics:**
- `rejection_rate`: Fraction of connection attempts that fail due to capacity
- `saturation_frac`: Fraction of agents at their capacity limit

**Impact on Results:**
- Delays percolation by limiting high-degree nodes
- Creates more uniform degree distributions
- Introduces bottleneck effects in network formation

### 4. Improvement 3: External Shocks (`improvement3_shocks.py`)

Models **temporal variations** in connection formation rates due to external events.

**New Features:**
- **Shock Events**: Temporary increases in connection formation rate
- **Time-Varying Dynamics**: Connection rate multipliers at specific time steps
- **Custom Rate Schedules**: Arbitrary rate patterns over time

**Key Parameters:**
- `shock_steps`: Time steps when shocks occur
- `shock_multiplier`: Factor by which connection rate increases during shocks
- `shock_duration`: Number of steps each shock lasts
- `rate_schedule`: Optional custom rate pattern for all time steps

**Impact on Results:**
- Creates accelerated percolation during shock periods
- Demonstrates sensitivity to timing of external events
- Shows how external perturbations affect network formation

### 5. Realistic Model (`realistic_button_network.py`)

**Combines all improvements** with toggle switches for comprehensive analysis.

**Toggle Features:**
- `use_heterogeneity`: Enable/disable agent diversity and homophily
- `use_capacity`: Enable/disable capacity constraints
- `use_shocks`: Enable/disable external shock events

**Benefits:**
- Allows systematic comparison of individual vs. combined effects
- Provides most realistic representation of complex network formation
- Enables sensitivity analysis of different mechanisms

## Analysis and Results

### Percolation Threshold Analysis

The key metric is the **percolation threshold** - the threads/buttons ratio where the giant component first reaches 50% of all nodes. This threshold indicates:

- **Network Efficiency**: Lower thresholds mean faster connectivity
- **Robustness**: How external factors affect network formation
- **Critical Points**: Where small changes have large effects

### Output Visualization

The generated plot shows:

- **X-axis**: Threads/Buttons ratio (edge density)
- **Y-axis**: Giant component fraction (largest connected component size)
- **Curves**: Each model's percolation behavior
- **Annotations**: Threshold points where giant component reaches 50%
- **Horizontal Line**: 50% threshold reference

### Interpretation of Results

From the provided visualization:

1. **Baseline (Blue)**: Shows classic percolation with smooth S-curve transition
2. **Hetero+Homophily (Orange)**: May show accelerated or delayed percolation depending on group structure
3. **Capacity (Green)**: Typically shows delayed percolation due to connection limitations
4. **Shocks (Red)**: Shows stepped increases during shock events
5. **All Toggles (Purple)**: Combines all effects, showing most complex dynamics

### Key Observations

- **Phase Transition**: All models show rapid transition from disconnected to connected
- **Threshold Variation**: Different mechanisms shift the critical threshold
- **Non-monotonic Effects**: Some improvements may increase threshold (slower percolation)
- **Realistic Complexity**: Combined model shows most realistic network formation

## Running the Code

### Prerequisites

```bash
pip install -r requirements.txt
```

### Execution

```bash
python run_test.py
```

This will:
1. Run all five model variants with 20 iterations each
2. Calculate percolation thresholds for each model
3. Generate comparative visualization
4. Print threshold analysis summary

### Key Parameters

Modify parameters in `run_test.py`:

```python
BASE = dict(n=2000, steps=40, speed=0.03)  # Basic simulation settings
HETERO = dict(groups=3, activity_mu=-1.0, activity_sigma=0.8, homophily=0.75)  # Social parameters
CAPACITY = dict(capacity_mu=12, capacity_sigma=0)  # Capacity constraints
SHOCKS = dict(shock_steps=(8, 20), shock_multiplier=3.0, shock_duration=2)  # Shock events
ITERATIONS = 20  # Number of simulation runs for averaging
```

## Scientific Significance

This project demonstrates:

1. **Complex Systems Modeling**: How simple rules create emergent collective behavior
2. **Phase Transitions**: Critical phenomena in network formation
3. **Multi-factor Analysis**: Systematic study of interacting mechanisms
4. **Social Network Theory**: Realistic modeling of human connection patterns
5. **Robustness Testing**: How external shocks affect system behavior

The Button Network serves as a powerful metaphor for understanding connectivity in social networks, infrastructure systems, epidemic spread, and other complex networks where percolation phenomena are crucial.

## Technical Details

- **Framework**: AgentPy for agent-based modeling
- **Network Analysis**: NetworkX for graph algorithms
- **Statistics**: NumPy/Pandas for data analysis
- **Visualization**: Matplotlib for plotting
- **Reproducibility**: Seeded random number generation ensures consistent results
