.PHONY: install install-dev test verify-benchmark validate help

help:
	@echo "install           install pinned dependencies + this package (editable)"
	@echo "install-dev       install + dev tools"
	@echo "test              run the offline test suite"
	@echo "verify-benchmark  check outputs/final_benchmark is unchanged (local only)"
	@echo "validate          validate data artifacts"

install:
	pip install -r requirements.txt
	pip install -e . --no-deps

install-dev: install
	pip install -e ".[dev]"

test:
	python -m pytest -q

verify-benchmark:
	python tools/benchmark_manifest.py verify

validate:
	rag-thesis validate
