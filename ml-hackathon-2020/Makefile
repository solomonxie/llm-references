VENV := venv
PY := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

.PHONY: venv install run clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install pandas scikit-learn xgboost joblib

# Run a submission script, e.g. `make run N=7`
run:
	$(PY) submit$(N).py

clean:
	rm -f model_submit*.joblib
	rm -rf $(VENV)
