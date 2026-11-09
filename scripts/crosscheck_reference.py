"""Cross-check CausalForge estimators against EconML / DoWhy (run where they are installed).

    pip install -e ".[reference]" && PYTHONPATH=src python scripts/crosscheck_reference.py

NOTE: written without those libraries available in the development sandbox, so this script
has NOT been executed yet. If an API call differs in your installed version, adjust it and
record the versions used. Agreement is expected within Monte-Carlo/estimator differences,
not bit-for-bit.
"""

import numpy as np

from causalforge.estimators.doubly_robust import aipw_ate
from causalforge.estimators.regression import regression_adjustment
from causalforge.synthetic.generators import make_scm

ds = make_scm(5000, 1)
print(
    f"truth {ds.true_ate:.3f} | ours: RA {regression_adjustment(ds).estimate:.3f}, AIPW {aipw_ate(ds).estimate:.3f}"
)
try:
    import pandas as pd
    from dowhy import CausalModel

    df = pd.DataFrame(ds.X, columns=ds.feature_names).assign(T=ds.treatment, Y=ds.outcome)
    m = CausalModel(df, treatment="T", outcome="Y", common_causes=ds.feature_names)
    est = m.estimate_effect(m.identify_effect(), method_name="backdoor.linear_regression")
    print("DoWhy backdoor.linear_regression:", est.value)
except Exception as e:  # noqa: BLE001
    print("DoWhy check failed/unavailable:", repr(e))
try:
    from econml.dml import CausalForestDML, LinearDML
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    kw = dict(
        model_y=GradientBoostingRegressor(),
        model_t=GradientBoostingClassifier(),
        discrete_treatment=True,
    )
    l = LinearDML(**kw).fit(ds.outcome, ds.treatment, X=None, W=ds.X)
    print("EconML LinearDML ATE:", float(np.ravel(l.const_marginal_ate())[0]))
    cf = CausalForestDML(**kw, n_estimators=200).fit(ds.outcome, ds.treatment, X=ds.X)
    print("EconML CausalForestDML ATE:", float(np.ravel(cf.ate(ds.X))[0]))
except Exception as e:  # noqa: BLE001
    print("EconML check failed/unavailable:", repr(e))
