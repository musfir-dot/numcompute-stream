
from __future__ import annotations

import time

import numpy as np

from numcompute_stream.stats import RunningMoments


def naive_loop_moments(chunks):
    
    n = 0
    mean = None
    m2 = None
    for chunk in chunks:
        for row in chunk:
            if mean is None:
                mean = np.zeros_like(row, dtype=float)
                m2 = np.zeros_like(row, dtype=float)
            n += 1
            delta = row - mean
            mean = mean + delta / n
            m2 = m2 + delta * (row - mean)
    var = m2 / n
    return mean, var


def vectorised_moments(chunks):
    rm = RunningMoments()
    for chunk in chunks:
        rm.update(chunk)
    return rm.mean_, rm.variance_


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{'n_samples':>10} | {'loop (s)':>9} | {'vector (s)':>11} | {'speed-up':>9}")
    print("-" * 49)
    for n in (5000, 20000, 50000):
        data = rng.normal(size=(n, 20))
        chunks = np.array_split(data, 50)

        t0 = time.perf_counter()
        m_loop, v_loop = naive_loop_moments(chunks)
        t_loop = time.perf_counter() - t0

        t0 = time.perf_counter()
        m_vec, v_vec = vectorised_moments(chunks)
        t_vec = time.perf_counter() - t0

    
        assert np.allclose(m_loop, m_vec, atol=1e-9)
        assert np.allclose(m_vec, data.mean(axis=0), atol=1e-9)

        speed = t_loop / t_vec if t_vec > 0 else float("inf")
        print(f"{n:>10} | {t_loop:>9.4f} | {t_vec:>11.4f} | {speed:>8.1f}x")


if __name__ == "__main__":
    main()
