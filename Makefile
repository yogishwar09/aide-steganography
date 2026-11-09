PY ?= .venv/bin/python
AIDE ?= .venv/bin/aide

.PHONY: install data train detector test lint fmt bench web docker deploy report

install:            ## create venv-installed package with all extras
	$(PY) -m pip install -e ".[dev,web,data,report]"

data:               ## fetch BOSSbase sample covers
	$(PY) scripts/fetch_dataset.py --count 300

train:              ## train the AIDE decision engine
	$(AIDE) train --images data/samples --out data/models

detector:           ## train the ML steganalyser
	$(AIDE) train-detector --images data/samples --out data/models

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check .

fmt:
	.venv/bin/ruff check . --fix

bench:              ## full robustness benchmark -> reports/benchmark
	$(AIDE) benchmark --num-images 30 --out reports/benchmark

web:
	$(PY) webapp/app.py

docker:
	docker build -t aide .

deploy:             ## publish the console to a Hugging Face Space (see scripts/deploy_space.py)
	$(PY) scripts/deploy_space.py
