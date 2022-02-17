.PHONY: lint
lint:
	poetry run black main.py local_test.py

.PHONY: test
test:
	poetry run python local_test.py --count 10

.PHONY: test_all
test_all:
	poetry run python local_test.py --count 1000 --disable_tqdm
