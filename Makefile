.PHONY: build

build:
	python -m pip install -e .
	python dziko.py
