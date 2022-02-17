.PHONY: lint
lint:
	black main.py local_test.py

.PHONY: test
test:
	python local_test.py 10

.PHONY: test_all
test_all:
	python local_test.py 1000
