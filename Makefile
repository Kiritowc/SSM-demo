.PHONY: install demo up down train-cv train-ets smoke

SSM_ROOT := $(CURDIR)
export SSM_ROOT

PY ?= /home/sunshink/miniconda3/envs/ssdet/bin/python

install:
	$(PY) -m pip install -e . --no-deps

demo:
	$(PY) -m ssm.cli demo

up:
	$(PY) -m ssm.cli up

down:
	$(PY) -m ssm.cli down

smoke:
	bash scripts/dev/smoke_test.sh

train-cv:
	$(PY) -m ssm.cli train cv

train-ets:
	$(PY) -m ssm.cli train ets --model ets_a --data scada
