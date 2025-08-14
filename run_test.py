# run_test.py
# Runs baseline & improved Button Network models and plots a combined comparison curve.
# Robust across AgentPy builds (handles long/wide dataframes; no hard dependency on .runtime).

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

# Heterogeneity params
HETERO = dict(groups=3, activity_mu=-1.0, activity_sigma=0.8, homophily=0.75)

# Capacity params
CAPACITY = dict(capacity_mu=12, capacity_sigma=0)  # try smaller mu to make it tighter

# Shocks params (tuples, not lists -> hashable for pandas/AgentPy)
SHOCKS = dict(shock_steps=(8, 20), shock_multiplier=3.0, shock_duration=2)

ITERATIONS = 20


# --------- helpers ---------
def run_exp(model_cls, params, iterations=ITERATIONS):
    print("Scheduled runs:", iterations)
    sample = ap.Sample(params)
    exp = ap.Experiment(model_cls, sample, iterations=iterations, record=True)
    results = exp.run()
    print("Experiment finished")
    # Some AgentPy builds don't expose .runtime; print only if we can find it.
    try:
        rt = getattr(results, "runtime", None)
        if rt is None and hasattr(results, "reporters"):
            # sometimes stored as a reporter
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
    """Return (x, y) averaged over iterations for this AgentPy results object.
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

def _threshold_from_curve(x, y, target=0.5):
    """Return (x*, y*) where the curve first reaches target (linear interpolation)."""
    import numpy as np
    xv = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
    yv = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    idx = (yv >= target).nonzero()[0]
    if len(idx) == 0:
        return float('nan'), float('nan')
    i = idx[0]
    if i == 0:
        return float(xv[0]), float(yv[0])
    # linear interpolate between (i-1) and i
    x0, x1 = xv[i-1], xv[i]
    y0, y1 = yv[i-1], yv[i]
    if y1 == y0:
        return float(x1), float(y1)
    x_star = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return float(x_star), float(target)


def plot_all_curves_annotated(results_dict, title="Giant component vs Threads/Buttons"):
    """Overlay curves and annotate threshold (giant_frac=0.5) for each model."""
    import matplotlib.pyplot as plt

    plt.figure()
    threshold_rows = []  # for printing a quick table after the plot

    for label, res in results_dict.items():
        x, y = _extract_xy(res)
        plt.plot(x, y, label=label)

        # Annotate threshold from the averaged curve
        x_star, y_star = _threshold_from_curve(x, y, target=0.5)
        if x_star == x_star:  # not NaN
            plt.scatter([x_star], [y_star], s=35, marker='o')
            plt.annotate(f'{label}\n≈ {x_star:.3f}',
                         xy=(x_star, y_star),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8)
        # Also pull the run-reported mean threshold if available
        import pandas as pd
        thr_report = pd.to_numeric(res.reporters.get('threshold_t_over_b'), errors='coerce').mean()
        threshold_rows.append((label, thr_report, x_star))

    plt.axhline(0.5, ls='--', lw=1, alpha=0.6)  # target line
    plt.xlabel('Threads / Buttons')
    plt.ylabel('Largest component (fraction)')
    plt.title(title + ' (threshold = 0.5)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print a compact comparison table (reported vs from-curve)
    print("\nEstimated thresholds (threads/buttons):")
    for label, rep, curve in threshold_rows:
        rep_txt = "NaN" if pd.isna(rep) else f"{rep:.3f}"
        curve_txt = "NaN" if curve != curve else f"{curve:.3f}"
        print(f"- {label:16s} reported ≈ {rep_txt} | curve ≈ {curve_txt}")

def plot_all_curves(results_dict, title="Giant component vs Threads/Buttons"):
    plt.figure()
    for label, res in results_dict.items():
        x, y = _extract_xy(res)
        plt.plot(x, y, label=label)
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

    # Heterogeneity + Homophily
    P1 = dict(BASE, **HETERO)
    res_i1 = run_exp(ButtonHeteroHomophily, P1, iterations=ITERATIONS)
    summarize_threshold(res_i1, 'Hetero+Homophily')

    # Capacity constraints
    P2 = dict(BASE, **CAPACITY)
    res_i2 = run_exp(ButtonWithCapacity, P2, iterations=ITERATIONS)
    summarize_threshold(res_i2, 'Capacity')

    # Shocks only
    P3 = dict(BASE, **SHOCKS)
    res_i3 = run_exp(ButtonWithShocks, P3, iterations=ITERATIONS)
    summarize_threshold(res_i3, 'Shocks')

    # All toggles combined
    PALL = dict(
        BASE,
        use_heterogeneity=True, **HETERO,
        use_capacity=True, **CAPACITY,
        use_shocks=True, **SHOCKS
    )
    res_all = run_exp(RealisticButtonModel, PALL, iterations=ITERATIONS)
    summarize_threshold(res_all, 'All-toggles')

    # Combined comparison plot (overlay all curves)
    plot_all_curves_annotated({
    'Baseline': res_base,
    'Hetero+Homophily': res_i1,
    'Capacity': res_i2,
    'Shocks': res_i3,
    'All-toggles': res_all
    })

