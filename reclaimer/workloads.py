"""Synthetic CPU-demand generators (millicores per 15 s reading) for controlled experiments.

Level (low/high) and temporal behaviour (constant/ramp/periodic/bursty/irregular) are
separate axes in principle; each family below fixes one combination.  Parameters are
re-drawn from `seed`, so different seeds are different traces of the same family.
Demand may exceed the original 1000 m allocation (see `high`, `bursty`), i.e. some
throttling is unavoidable even for static allocation.
"""
from __future__ import annotations
import numpy as np

N = 160


def _fin(x):  return np.clip(np.round(x), 0, None)


def constant(rng, n=N):  return _fin(rng.uniform(300, 500) + rng.normal(0, 15, n))


def low(rng, n=N):
    x = rng.uniform(80, 150) + rng.normal(0, 10, n)
    b = rng.random(n) < 0.05
    x[b] += rng.uniform(40, 120, b.sum())
    return _fin(x)


def high(rng, n=N):  return _fin(np.clip(rng.uniform(780, 880) + rng.normal(0, 80, n), 0, 1300))


def ramp(rng, n=N):
    return _fin(np.linspace(rng.uniform(100, 200), rng.uniform(850, 1000), n) + rng.normal(0, 30, n))


def periodic(rng, n=N):
    per = int(rng.integers(12, 24)); ph = rng.uniform(0, per)
    on = (((np.arange(n) + ph) % per) / per) < 0.4
    return _fin(np.where(on, rng.uniform(600, 800), rng.uniform(80, 150)) + rng.normal(0, 25, n))


def bursty(rng, n=N):
    x = rng.uniform(100, 200) + rng.normal(0, 15, n)
    for t in range(n):
        if rng.random() < 0.08:
            d = int(rng.integers(1, 4)); x[t:t + d] = np.maximum(x[t:t + d], rng.uniform(600, 1150))
    return _fin(x)


def irregular(rng, n=N):
    x, t = np.zeros(n), 0
    while t < n:
        d = int(rng.geometric(1 / 25)) + 5; lvl = rng.choice([150, 350, 600, 850])
        x[t:t + d] = lvl; t += d
    return _fin(x + rng.normal(0, 40, n))


FAMILIES = {"constant": constant, "low": low, "high": high, "ramp": ramp,
            "periodic": periodic, "bursty": bursty, "irregular": irregular}


def recorded_like(rng, series: np.ndarray, n=N, block=6):
    """Block bootstrap of the *recorded* trace (derived from real data, not independent of it)."""
    out = []
    while len(out) < n:
        i = int(rng.integers(0, len(series) - block)); out.extend(series[i:i + block])
    return np.asarray(out[:n], float)


def make(family: str, seed: int, recorded: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(10_000 + seed)
    return recorded_like(rng, recorded) if family == "recorded-like" else FAMILIES[family](rng)
