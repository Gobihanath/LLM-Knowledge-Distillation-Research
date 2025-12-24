
# Unified Reasoning Benchmark

This repository provides a unified evaluation framework for benchmarking reasoning capabilities of LLMs on multiple datasets.

## Structure
- `datasets/`: Raw datasets (jsonl format)
- `scripts/`: Processing, running, and evaluation logic
- `results/`: Outputs in CSV

## Usage
```bash
cd scripts
python preprocess.py
python benchmark_runner.py
```

## Output
CSV file with all dataset results saved in `results/benchmark_results.csv`


https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/blob/main/am_0.9M_sample_1k.jsonl.zst
