# run_test.py
# Runs baseline & improved Button Network models and opens:
# - one window per model (same color per model everywhere)
# - one combined window overlaying all curves
# Robust to AgentPy arrange_variables() returning "long" or "wide" frames.

import agentpy as ap
import matplotlib.pyplot as plt
import pandas as pd

from baseline_button_network import BaselineButtonModel
from improvement1_hetero import ButtonHeteroHomophily
from improvement2_capacity import ButtonWithCapacity
from improvement3_shocks import ButtonWithShocks
from realistic_button_network import RealisticButtonModel

# --------- parameters ---------
BASE = dict(n=2000, steps=40, speed=0.03)
HETERO = dict(groups=3, activity_mu=-1.0, activity_sigma=0.8, homophily=0.75)
CAPACITY = dict(capacity_mu=12, capacity_sigma=0)
SHOCKS = dict(shock_steps=(8, 20), shock_multiplier=3.0, shock_duration=2)
ITERATIONS = 20

# Consistent colors across all plots
MODEL_COLORS = {
    "Baseline":            "#1f77b4",  # blue
    "Hetero+Homophily":    "#ff7f0e",  # orange
    "Capacity":            "#2ca02c",  # green
    "Shocks":              "#d62728",  # red
    "All-toggles":         "#9467bd",  # purple
}

# --------- helpers ---------
def run_exp(model_cls, params, iterations=ITERATIONS):
    print("Scheduled runs:", iterations)
    sample = ap.Sample(params)
    exp = ap.Experiment(model_cls, sample, iterations=iterations, record=True)
    results = exp.run()
    print("Experiment finished")
    # Print runtime only if available in this AgentPy build
    try:
        rt = getattr(results, "runtime", None)
        if rt is None and hasattr(results, "reporters"):
            rt = results.reporters.get("runtime", None)
        if rt is None and hasattr(results, "info") and isinstance(results.info, dict):
            rt = results.info.get("runtime", None)
        if rt is not None:
            print("Run time:", rt)
    except Exception:
        pass
    return results

def summarize_threshold(results, label):
    df = results.reporters
    thr = pd.to_numeric(df.get('threshold_t_over_b'), errors='coerce').mean()
    out = {'model': label, 'mean_threshold_t_over_b': thr}
    print(out)
    return out

def _extract_xy(results):
    """Return (x, y) averaged over iterations.
    Handles both LONG ('variable','value') and WIDE formats from arrange_variables()."""
    df = results.arrange_variables()

    if {'variable', 'value'}.issubset(df.columns):
        # LONG format
        df = df[df['variable'].isin(['giant_frac', 'threads_to_button'])].copy()
        grp = df.groupby(['t', 'variable'], as_index=False)['value'].mean()
        wide = grp.pivot(index='t', columns='variable', values='value').reset_index()
        x = wide['threads_to_button']
        y = wide['giant_frac']
    else:
        # WIDE format
        cols = [c for c in ['t', 'giant_frac', 'threads_to_button'] if c in df.columns]
        df_small = df[cols].copy()
        if 't' in df_small.columns:
            wide = df_small.groupby('t', as_index=False).mean(numeric_only=True)
        else:
            wide = df_small.mean(numeric_only=True).to_frame().T
            wide['t'] = range(len(wide))
        x = wide['threads_to_button']
        y = wide['giant_frac']
    return x, y

def plot_single_curve(label, results):
    """Open a window with one curve, colored consistently and titled per improvement."""
    color = MODEL_COLORS.get(label, None)
    x, y = _extract_xy(results)
    plt.figure()
    plt.plot(x, y, label=label, color=color)
    plt.xlabel('Threads / Buttons')
    plt.ylabel('Largest component (fraction)')
    plt.title(f'Giant component vs Threads/Buttons — {label}')
    plt.tight_layout()
    plt.show(block=False)  # show but don't block subsequent windows

def plot_all_curves(results_dict, title="Giant component vs Threads/Buttons"):
    """Open a window with all curves overlaid, colors matching single plots."""
    plt.figure()
    for label, res in results_dict.items():
        color = MODEL_COLORS.get(label, None)
        x, y = _extract_xy(res)
        plt.plot(x, y, label=label, color=color)
    plt.xlabel('Threads / Buttons')
    plt.ylabel('Largest component (fraction)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------- main ---------
if __name__ == "__main__":
    # Baseline
    res_base = run_exp(BaselineButtonModel, BASE, iterations=ITERATIONS)
    summarize_threshold(res_base, 'Baseline')
    plot_single_curve('Baseline', res_base)

    # Heterogeneity + Homophily
    P1 = dict(BASE, **HETERO)
    res_i1 = run_exp(ButtonHeteroHomophily, P1, iterations=ITERATIONS)
    summarize_threshold(res_i1, 'Hetero+Homophily')
    plot_single_curve('Hetero+Homophily', res_i1)

    # Capacity constraints
    P2 = dict(BASE, **CAPACITY)
    res_i2 = run_exp(ButtonWithCapacity, P2, iterations=ITERATIONS)
    summarize_threshold(res_i2, 'Capacity')
    plot_single_curve('Capacity', res_i2)

    # Shocks only
    P3 = dict(BASE, **SHOCKS)
    res_i3 = run_exp(ButtonWithShocks, P3, iterations=ITERATIONS)
    summarize_threshold(res_i3, 'Shocks')
    plot_single_curve('Shocks', res_i3)

    # All toggles combined
    PALL = dict(
        BASE,
        use_heterogeneity=True, **HETERO,
        use_capacity=True, **CAPACITY,
        use_shocks=True, **SHOCKS
    )
    res_all = run_exp(RealisticButtonModel, PALL, iterations=ITERATIONS)
    summarize_threshold(res_all, 'All-toggles')
    plot_single_curve('All-toggles', res_all)

    # Combined comparison (last window)
    plot_all_curves({
        'Baseline': res_base,
        'Hetero+Homophily': res_i1,
        'Capacity': res_i2,
        'Shocks': res_i3,
        'All-toggles': res_all
    })
