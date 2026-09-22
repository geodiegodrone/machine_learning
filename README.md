# Machine Learning Coursework Portfolio

**Curated machine-learning notebooks with reproducible project structure**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](requirements.txt)
[![CI](https://img.shields.io/badge/CI-repository%20checks-green)](.github/workflows/ci.yml)
[![Portfolio](https://img.shields.io/badge/status-curated%20academic%20portfolio-yellow)](docs/PORTFOLIO_GUIDE.md)

This repository collects machine-learning coursework and experiments. It has been upgraded from a notebook archive into a portfolio-ready academic repository with a project index, reproducibility checklist, dependency file, lightweight validation tests, and CI.

The `projects/` directory adds five end-to-end case studies covering customer segmentation, demand and energy forecasting, marketing response, and text classification. Each project documents its business framing, data provenance, evaluation design, limitations, and MLflow experiment tracking.

## Contents

| Path | Topic | Notes |
|---|---|---|
| `actividad1/` | Dengue prediction practice | Includes train/test feature CSVs and labels |
| `07MBID_Práctica_2_diego_pulido.ipynb` | ML practice 2 | Notebook experiment |
| `actividad1/*.ipynb` | ML practice 1 variants | Original notebook work |
| `projects/wholesale-customer-segmentation/` | Customer analytics | Unsupervised segmentation with tracked model selection |
| `projects/bike-demand-forecasting/` | Mobility | Leakage-aware hourly demand regression |
| `projects/bank-marketing-response/` | Marketing | Imbalanced response classification and threshold selection |
| `projects/home-energy-forecasting/` | Energy | Time-aware household energy forecasting |
| `projects/sms-spam-classifier/` | NLP | TF-IDF spam classification |

## Why This Repo Matters

Coursework repositories often become hard to review because they are only notebook dumps. This repository now explains what is inside, how to validate the data, and how it can evolve into a professional ML learning portfolio.

## Quickstart

```bash
python -m pip install -r requirements.txt
pytest -q
python scripts/catalog_repository.py
```

## Reproducibility Standards

- Keep raw coursework files unchanged.
- Add derived scripts under `scripts/`.
- Add documentation under `docs/`.
- Keep data files small and explicit.
- Avoid committing notebook checkpoints.
- Prefer deterministic random seeds in new notebooks.
- Track each experiment, parameters, metrics, and deployable model with MLflow.

## Next Upgrade Targets

- Convert notebooks into reproducible pipelines.
- Add model cards for each exercise.
- Export metrics tables to `reports/`.
- Add notebook execution checks.
- Add a unified `src/` package for reusable preprocessing.

## Author

Diego F. Pulido Sastoque
