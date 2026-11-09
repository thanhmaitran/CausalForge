"""Generate notebooks/*.ipynb from code cells (nbformat v4 JSON, no external dependency)."""

import json
from pathlib import Path

HDR = "import sys; sys.path.insert(0, '../src')\nimport numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
NB = {
    "01_dataset_exploration": (
        "Dataset exploration and data-quality audit",
        [
            HDR
            + "from causalforge.synthetic.generators import make_scm\nfrom causalforge.data.validators.causal_validator import validate_dataset\nds = make_scm(3000, 1)\nrep = validate_dataset(ds)\npd.Series({k: v for k, v in rep.items() if not isinstance(v, (dict, list))})",
            "# Real Criteo data (optional). exposure is post-treatment and is NEVER used as a covariate.\nimport os\nf = '../data/raw/criteo/criteo-research-uplift-v2.1.csv.gz'\nif os.path.exists(f):\n    from causalforge.data.loaders.criteo_stream import to_causal_dataset\n    df = pd.read_csv(f, nrows=200_000)\n    print(validate_dataset(to_causal_dataset(df))['treatment_ratio'])\nelse:\n    print('Criteo file not downloaded yet')",
        ],
    ),
    "02_causal_assumptions": (
        "DAGs, backdoor paths and adjustment sets",
        [
            HDR
            + "from causalforge.graphs.dag import CausalDAG\ng = CausalDAG()\nfor n, r in [('X','confounder'),('T','treatment'),('Y','outcome'),('M','mediator')]: g.add_variable(n, r)\nfor a, b in [('X','T'),('X','Y'),('T','M'),('M','Y')]: g.add_causal_edge(a, b)\nprint(g.backdoor_paths('T','Y'), g.adjustment_sets('T','Y'))\nprint('adjusting for mediator valid?', g.is_valid_adjustment_set({'M'},'T','Y'))"
        ],
    ),
    "03_propensity_analysis": (
        "Propensity scores: overlap and balance",
        [
            HDR
            + "from causalforge.synthetic.generators import make_scm\nfrom causalforge.estimators.propensity import *\nds = make_scm(3000, 2)\ne = estimate_propensity(ds.X, ds.treatment)[1]\nprint(overlap_report(e, ds.treatment))\nw = np.where(ds.treatment==1, 1/e, 1/(1-e))\nprint(np.abs(standardized_mean_difference(ds.X, ds.treatment)).max(), np.abs(standardized_mean_difference(ds.X, ds.treatment, w)).max())"
        ],
    ),
    "04_classical_estimators": (
        "Classical estimators on a confounded synthetic dataset",
        [
            HDR
            + "from causalforge.synthetic.generators import make_scm\nfrom causalforge.experiments.runner import CLASSICAL\nds = make_scm(3000, 3)\nrows = []\nfor k, f in CLASSICAL.items():\n    r = f(ds); rows.append((k, r.estimate, r.ci))\nprint('truth', ds.true_ate)\npd.DataFrame(rows, columns=['estimator','estimate','ci'])"
        ],
    ),
    "05_meta_learners": (
        "S/T/X learners",
        [
            HDR
            + "from causalforge.synthetic.generators import make_scm\nfrom causalforge.meta_learners.learners import *\nfrom causalforge.evaluation.metrics import pehe\nds = make_scm(3000, 4, randomized=True, nonlinear=True)\nfor L in (SLearner, TLearner, XLearner):\n    p = L().fit(ds.X, ds.treatment, ds.outcome).predict_cate(ds.X)\n    print(L.name, round(pehe(p, ds.true_cate), 3))"
        ],
    ),
    "06_double_ml": (
        "Cross-fitted DML and honest causal forest",
        [
            HDR
            + "from causalforge.synthetic.generators import make_scm\nfrom causalforge.causal_ml.dml import dml_plr\nfrom causalforge.causal_ml.causal_forest import HonestCausalForest\nds = make_scm(3000, 5, nonlinear=True, heterogeneous=False)\nr = dml_plr(ds); print('DML', r.estimate, r.ci, 'truth', ds.true_ate)\nf = HonestCausalForest().fit(ds.X, ds.treatment, ds.outcome); print('forest mean CATE', f.predict_cate(ds.X).mean())"
        ],
    ),
    "07_sensitivity_analysis": (
        "Sensitivity: robustness value, placebo, negative control",
        [
            HDR
            + "from causalforge.synthetic.generators import make_scm\nfrom causalforge.sensitivity.partial_r2 import sensitivity_report\nfrom causalforge.sensitivity.negative_controls import negative_control_outcome_test\nds = make_scm(3000, 6, hidden_t=1.0, hidden_y=1.0)\nprint(sensitivity_report(ds.X, ds.treatment, ds.outcome))\nnco = 0.8*ds.metadata['U'] + np.random.default_rng(0).normal(size=ds.n)\nprint(negative_control_outcome_test(ds, nco))"
        ],
    ),
    "08_policy_evaluation": (
        "Budget-constrained policy evaluation",
        [
            HDR
            + "from causalforge.synthetic.generators import make_scm\nfrom causalforge.meta_learners.learners import XLearner\nfrom causalforge.decision.policies import *\nds = make_scm(4000, 7, randomized=True, nonlinear=True)\ntr, te = ds.subset(np.arange(2000)), ds.subset(np.arange(2000, 4000))\nm = XLearner().fit(tr.X, tr.treatment, tr.outcome)\nfor b in (0.1, 0.3, 0.5):\n    pi = top_k_policy(m.predict_cate(te.X), b)\n    print(b, round(true_policy_value(te, pi), 3), round(ipw_policy_value(te.treatment, te.outcome, pi, te.propensity), 3))"
        ],
    ),
}
out = Path("notebooks")
out.mkdir(exist_ok=True)
for name, (title, cells) in NB.items():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "All numbers are computed when the notebook runs (outputs are cleared in git).",
                ],
            }
        ]
        + [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": c.splitlines(True),
            }
            for c in cells
        ],
    }
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote", len(NB))
