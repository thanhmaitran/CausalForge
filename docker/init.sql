-- PostgreSQL DDL (metadata only; raw datasets never go in the database)
CREATE TABLE datasets(id SERIAL PRIMARY KEY, name TEXT NOT NULL, version TEXT, source TEXT,
  checksum TEXT, license TEXT, created_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE causal_graphs(id SERIAL PRIMARY KEY, dataset_id INT REFERENCES datasets(id),
  graph_json JSONB NOT NULL, treatment_node TEXT, outcome_node TEXT, created_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE experiments(id SERIAL PRIMARY KEY, dataset_id INT REFERENCES datasets(id),
  estimator TEXT, estimand TEXT, configuration_json JSONB, random_seed INT, status TEXT,
  created_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE estimation_results(id SERIAL PRIMARY KEY, experiment_id INT REFERENCES experiments(id),
  ate DOUBLE PRECISION, att DOUBLE PRECISION, standard_error DOUBLE PRECISION,
  ci_lower DOUBLE PRECISION, ci_upper DOUBLE PRECISION, metrics_json JSONB);
CREATE TABLE sensitivity_results(id SERIAL PRIMARY KEY, experiment_id INT REFERENCES experiments(id),
  method TEXT, assumptions_json JSONB, results_json JSONB);
CREATE TABLE reports(id SERIAL PRIMARY KEY, experiment_id INT REFERENCES experiments(id),
  file_path TEXT, created_at TIMESTAMPTZ DEFAULT now());
