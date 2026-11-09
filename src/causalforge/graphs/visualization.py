"""DAG plotting."""

from __future__ import annotations

import matplotlib
import networkx as nx

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROLE_COLOR = {
    "treatment": "#ff7f0e",
    "outcome": "#2ca02c",
    "confounder": "#d62728",
    "mediator": "#9467bd",
    "collider": "#8c564b",
    "unobserved": "#bbbbbb",
    "instrument": "#17becf",
    "covariate": "#1f77b4",
}


def plot_dag(
    dag, path: str, title: str = "", highlight: set[str] | None = None, pos: dict | None = None
):
    g = dag.graph
    pos = pos or (nx.spring_layout(g, seed=3) if len(g) > 4 else nx.shell_layout(g))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    cols = [ROLE_COLOR.get(d.get("role"), "#1f77b4") for _, d in g.nodes(data=True)]
    lw = [3 if highlight and n in highlight else 0.8 for n in g]
    nx.draw_networkx(
        g,
        pos,
        ax=ax,
        node_color=cols,
        node_size=1100,
        font_color="white",
        edgecolors="k",
        linewidths=lw,
        arrowsize=18,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
