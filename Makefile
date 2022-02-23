.PHONY: lint
lint:
	poetry run black main.py local_test.py

.PHONY: test_with_count
test_with_count:
	poetry run python local_test.py --count $(COUNT)

.PHONY: test_with_id
test_with_id:
	poetry run python local_test.py --id $(ID)

.PHONY: test
test:
	poetry run python local_test.py --count 10

# TODO: PyPyでtest実行
.PHONY: test_nohints
test_nohints:
	poetry run python local_test.py --count 10 --no_hints

.PHONY: test_all
test_all:
	poetry run python local_test.py --count 1000 --disable_tqdm --process 20


.PHONY: strip_hints
strip_hints:
	poetry run strip-hints main.py | sed -e '/from __future__ import annotations/d' -e '/^from typing import/d' > main_nohints.py
	poetry run black main_nohints.py

.PHONY: jupyter
jupyter:
	poetry run jupyter lab
