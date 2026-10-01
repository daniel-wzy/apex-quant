.PHONY: test synth lint clean

test:
	python3 -m pytest tests/ -v

synth:
	python3 data/synthetic/generate.py
	python3 quant/gen_sim_trades.py \
		--input data/synthetic/trades_sample.csv \
		--output data/synthetic/sim_out.csv \
		--ohlcv data/synthetic/ohlcv_sample.csv

lint:
	python3 -m py_compile quant/*.py tests/*.py indicators/*.py

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
