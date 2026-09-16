.PHONY: help test db-up db-down smoke ingest features train evaluate app clean

PY := .venv/bin/python

help:
	@echo "test      - run the test suite"
	@echo "db-up     - start the Postgres container"
	@echo "db-down   - stop the Postgres container (add ARGS=-v to wipe data)"
	@echo "smoke     - build + run the full pipeline in Docker (few seasons)"
	@echo "ingest    - load all seasons 2010-2024 into Postgres"
	@echo "features  - build the model_frame table"
	@echo "train     - train models, report val Brier"
	@echo "evaluate  - evaluate + calibrate on test, write reliability plots"
	@echo "app       - launch the Streamlit frontend (needs: pip install -r requirements-app.txt)"

test:
	$(PY) -m pytest tests/ -q

db-up:
	docker compose up -d db

db-down:
	docker compose down $(ARGS)

smoke:
	docker compose up --build --abort-on-container-exit app

ingest:
	$(PY) -m presnap.ingest --seasons 2010:2024

features:
	$(PY) -m presnap.build_features

train:
	$(PY) -m presnap.train

evaluate:
	$(PY) -m presnap.evaluate

app:
	$(PY) -m streamlit run app/Home.py

clean:
	rm -rf artifacts reports
