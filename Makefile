.PHONY: up down

SSM_ROOT := $(CURDIR)
export SSM_ROOT
export LLAMA_SERVER?=$(HOME)/llama.cpp/build/bin/llama-server

PY ?= $(shell if [ -x "$(HOME)/miniconda3/envs/ssm/bin/python" ]; then echo "$(HOME)/miniconda3/envs/ssm/bin/python"; else command -v python3; fi)

up:
	$(PY) -m ssm.cli up

down:
	$(PY) -m ssm.cli down
