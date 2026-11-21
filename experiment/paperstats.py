"""Shared statistics for every number the paper prints.

Both make_paper_tables.py and stats.py import from here, so a quantity has exactly one
definition. In particular the bootstrap is seeded *per call*: the same input array
always yields the same interval, regardless of which table asks for it first. (The
submitted version drew from one shared RNG, so the same baseline group printed as
[+2.35,+3.15], [+2.36,+3.14] and [+2.36,+3.15] in different places.)
"""
import math

import numpy as np
from scipy import stats

NBOOT = 20000
SEED = 0


def bias(rows):
    """Per-prompt pronoun log-odds b = log P(' he') - log P(' she')."""
    return np.array([math.log(r["p_he"]) - math.log(r["p_she"]) for r in rows])


def ci(x, n=NBOOT, alpha=0.05):
    """Percentile bootstrap CI of the mean. Deterministic in x."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    d = x[idx].mean(axis=1)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def wilcoxon_p(b0, b1):
    d = np.asarray(b1) - np.asarray(b0)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(b1, b0, alternative="two-sided", zero_method="wilcox").pvalue)


def paired(b0, b1):
    """(mean diff, CI of mean diff, uncorrected Wilcoxon p, Cohen's d_z)."""
    b0, b1 = np.asarray(b0), np.asarray(b1)
    if b0.shape != b1.shape or b0.size == 0:
        return float("nan"), (float("nan"), float("nan")), 1.0, float("nan")
    d = b1 - b0
    if np.allclose(d, 0):
        return 0.0, (0.0, 0.0), 1.0, 0.0
    lo, hi = ci(d)
    return float(d.mean()), (lo, hi), wilcoxon_p(b0, b1), float(d.mean() / (d.std(ddof=1) + 1e-12))


def holm(pvals):
    """Holm step-down adjusted p-values, same order as the input."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def fmt_p(p):
    if p >= 0.05:
        return "n.s."
    if p <= 5e-4:
        return r"$\le$4.9e-4"
    return f"{p:.1e}".replace("e-0", "e-")


def fmt_b(x):
    return f"${x:+.2f}$"


def fmt_ci(lo, hi):
    return f"$[{lo:+.2f},{hi:+.2f}]$"
