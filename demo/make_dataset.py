
from __future__ import annotations

import csv
import os

import numpy as np


def main(path: str = None, n: int = 8000, seed: int = 7) -> None:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "stream_data.csv")
    rng = np.random.default_rng(seed)

    age = rng.uniform(18, 70, size=n)
    income = rng.normal(60000, 18000, size=n).clip(8000, None)
    score = rng.normal(0, 1, size=n)
    balance = rng.normal(5000, 3000, size=n)


    regions = np.array(["north", "south", "east", "west"])
    region = regions[rng.integers(0, 4, size=n)]
    region_effect = {"north": 0.4, "south": -0.3, "east": 0.1, "west": -0.2}
    reg_val = np.array([region_effect[r] for r in region])


    logit = (
        0.00012 * (income - 60000)
        + 1.6 * score * (age > 40)
        - 0.00008 * (balance - 5000)
        + 2.5 * reg_val
        + 1.3 * np.sin(score * 2)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    label = (rng.uniform(size=n) < prob).astype(int)

   
    income_str = [f"{v:.2f}" for v in income]
    score_str = [f"{v:.4f}" for v in score]
    miss_income = rng.uniform(size=n) < 0.02
    miss_score = rng.uniform(size=n) < 0.02
    for i in range(n):
        if miss_income[i]:
            income_str[i] = ""
        if miss_score[i]:
            score_str[i] = ""

    header = ["age", "income", "score", "balance", "region", "label"]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i in range(n):
            writer.writerow([
                f"{age[i]:.1f}", income_str[i], score_str[i],
                f"{balance[i]:.2f}", region[i], int(label[i]),
            ])
    print(f"Wrote {n} rows to {path}")


if __name__ == "__main__":
    main()
