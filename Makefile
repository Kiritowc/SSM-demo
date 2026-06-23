.PHONY: up down

SSM_ROOT := $(CURDIR)
export SSM_ROOT

PY ?= $(shell command -v python3)

up:
	$(PY) -m ssm.cli up

down:
	$(PY) -m ssm.cli down
