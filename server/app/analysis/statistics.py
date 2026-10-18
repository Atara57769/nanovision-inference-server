"""Summary statistics for per-particle measurements."""

import numpy as np


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "variance": None,
                "histogram": {"bin_edges": [], "counts": []}}
    data = np.asarray(values, dtype=float)
    counts, edges = np.histogram(data, bins=10)
    return {"count": len(values), "mean": float(data.mean()),
            "variance": float(data.var()),
            "histogram": {"bin_edges": edges.tolist(), "counts": counts.tolist()}}
