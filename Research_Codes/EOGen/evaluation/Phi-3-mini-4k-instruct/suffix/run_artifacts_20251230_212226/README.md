# Run provenance bundle

This folder captures the exact code, environment, and model snapshot fingerprints used for the experiments.

## Contents
- `code/` — Python scripts used for the run (snapshot at collection time)
- `pbs/` — PBS scripts used to submit array/merge jobs
- `env/`
  - `python_versions.txt` — Python / torch / transformers versions
  - `pip_freeze.txt` — full package list
  - `system.txt` — host + filesystem info
  - `gpu.txt` — GPU + driver info (if available)
  - `git.txt` — git commit + status (if applicable)
- `models/` — model snapshot fingerprints (SHA256 of weight shards + tokenizer/config files)
- `run_config.txt` — human-editable notes about the run configuration

## How results were produced
- Evaluation is parallelized with PBS arrays (`num_chunks` and `chunk_index` in the PBS scripts).
- Each array task writes a per-chunk CSV; a separate merge job concatenates chunks and generates the merged CSV/plots.

## Reproducing this run (high-level)
1. Use the same model snapshot:
   - If you use local snapshots, verify files using `models/*_fingerprint.txt`.
   - If you use HF ids, pin to the same revision/commit and confirm the same shard hashes.
2. Create a matching environment using `env/pip_freeze.txt` (and torch/transformers versions).
3. Run the PBS scripts in `pbs/` (or run locally with the same arguments and seeds).
4. Merge outputs using the merge script/mode recorded in `pbs/` or `code/`.

## Notes
- If resuming chunked runs, keep `num_chunks` constant to preserve the prompt?chunk mapping.
