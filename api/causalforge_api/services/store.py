"""Metadata store. SQLite (stdlib) for tests/local; docker/init.sql has the PostgreSQL DDL
with the same tables. Large artifacts stay on disk; only metadata/results are stored."""
from __future__ import annotations

import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments(id INTEGER PRIMARY KEY AUTOINCREMENT, estimator TEXT,
  estimand TEXT, configuration_json TEXT, random_seed INTEGER, status TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS estimation_results(id INTEGER PRIMARY KEY AUTOINCREMENT,
  experiment_id INTEGER, ate REAL, standard_error REAL, ci_lower REAL, ci_upper REAL,
  metrics_json TEXT);
CREATE TABLE IF NOT EXISTS causal_graphs(id INTEGER PRIMARY KEY AUTOINCREMENT, graph_json TEXT,
  treatment_node TEXT, outcome_node TEXT, created_at REAL);
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)

    def add_graph(self, graph_json, t, y) -> int:
        c = self.db.execute("INSERT INTO causal_graphs(graph_json,treatment_node,outcome_node,"
                            "created_at) VALUES(?,?,?,?)", (graph_json, t, y, time.time()))
        self.db.commit()
        return c.lastrowid

    def get_graph(self, gid: int) -> dict | None:
        r = self.db.execute("SELECT graph_json,treatment_node,outcome_node FROM causal_graphs "
                            "WHERE id=?", (gid,)).fetchone()
        return None if r is None else {"graph": json.loads(r[0]), "treatment": r[1], "outcome": r[2]}

    def add_experiment(self, estimator, estimand, config, seed, result: dict) -> int:
        c = self.db.execute("INSERT INTO experiments(estimator,estimand,configuration_json,"
                            "random_seed,status,created_at) VALUES(?,?,?,?,?,?)",
                            (estimator, estimand, json.dumps(config, default=str), seed,
                             "completed", time.time()))
        eid = c.lastrowid
        ci = result.get("ci") or (None, None)
        self.db.execute("INSERT INTO estimation_results(experiment_id,ate,standard_error,ci_lower,"
                        "ci_upper,metrics_json) VALUES(?,?,?,?,?,?)",
                        (eid, result["estimate"], result.get("se"), ci[0], ci[1],
                         json.dumps(result.get("diagnostics", {}), default=str)))
        self.db.commit()
        return eid

    def get_experiment(self, eid: int) -> dict | None:
        r = self.db.execute("SELECT e.id,e.estimator,e.estimand,e.status,r.ate,r.standard_error,"
                            "r.ci_lower,r.ci_upper,r.metrics_json FROM experiments e JOIN "
                            "estimation_results r ON r.experiment_id=e.id WHERE e.id=?", (eid,)).fetchone()
        if r is None:
            return None
        return dict(zip(["id", "estimator", "estimand", "status", "ate", "se", "ci_lower",
                         "ci_upper"], r[:8])) | {"diagnostics": json.loads(r[8])}

    def count_experiments(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
