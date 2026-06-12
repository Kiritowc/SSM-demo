.PHONY: install demo up down train-cv train-ets

SSM_ROOT := $(CURDIR)
export SSM_ROOT

PY ?= /home/sunshink/miniconda3/envs/ssdet/bin/python

install:
	$(PY) -m pip install -e packages/common -e packages/ets -e packages/vlm -e packages/cv --no-deps
	$(PY) -m pip install -e . --no-deps

demo:
	python -m ssm.cli demo

up:
	python -m ssm.cli up

down:
	python -m ssm.cli down

train-cv:
	python -m ssm.cli train cv

train-ets:
	python -m ssm.cli train ets --model ets_a --data scada
