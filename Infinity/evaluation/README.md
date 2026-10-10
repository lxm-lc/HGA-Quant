# Evaluation

The maintained evaluation entry points for this Infinity branch are:

```bash
bash scripts/run_dpg.sh
bash scripts/run_geneval.sh
bash scripts/run_imagereward.sh
```

These scripts run Infinity generation first and then launch the corresponding evaluator. They share common model, quantization, asset-download, and run-record logic through `scripts/common_assets.sh`.

Benchmark-specific dependencies are listed in:

```text
requirements-dpg.txt
requirements-geneval.txt
requirements-imagereward.txt
```

For full setup, required file paths, quantization environment variables, and evaluation-only mode, see the repository root `README.md`.
