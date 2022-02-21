.PHONY: lint
lint:
	poetry run black main.py local_test.py

.PHONY: test
test:
	poetry run python local_test.py --count 10

# .PHONY: test_nohints
# test_nohints:
# 	poetry run /Users/takizawa/.pyenv/versions/pypy3.8-7.3.7/bin/python local_test.py --count 10 --no_hints

.PHONY: test_all
test_all:
	poetry run python local_test.py --count 1000 --disable_tqdm

.PHONY: strip_hints
strip_hints:
	poetry run strip-hints main.py -o main_nohints.py

.PHONY: jupyter
jupyter:
	poetry run jupyter lab
