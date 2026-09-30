.PHONY: test synth lint clean

test:
	python -m pytest tests/ -v

synth:
	python data/synthetic/generate.py
	python quant/gen_sim_trades.py \
		--input data/synthetic/trades_sample.csv \
		--output data/synthetic/sim_out.csv \
		--ohlcv data/synthetic/ohlcv_sample.csv

lint:
	python -m py_compile quant/*.py tests/*.py indicators/*.py

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
