# Projet ML industrialisé

[![CI](https://github.com/Hirsman/ml-housing-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Hirsman/ml-housing-project/actions/workflows/ci.yml)

## Installation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Tests
pytest -v

## Qualité
ruff check .
black --check .
bandit -r src -ll
