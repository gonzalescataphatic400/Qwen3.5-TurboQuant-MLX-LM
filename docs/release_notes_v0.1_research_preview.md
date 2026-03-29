# TurboMLX v0.1 Research Preview

TurboMLX `v0.1 Research Preview` is a Qwen-first preview release for Apple Silicon MLX workflows.

## Supported Contract

- Qwen3 / Qwen3.5 full-attention layers using the default non-rotating `KVCache` path
- `baseline`, `mlx_quant`, and `turbomlx` generate / benchmark entrypoints
- `native_mlx` preview only for:
  - `mode=mse`
  - `bits_total=4`
  - `values_mode=dense`

## Preview Guarantees

- correctness-first preview behavior
- prompt-cache roundtrip via TurboMLX-owned save/load wrappers
- benchmark methodology with warmup runs, repeats, median reporting, and explicit `scorer_route`
- trusted-local-only prompt-cache warnings for `pickle`-backed cache files

## Known Limits

- `native_mlx` is a Stage A remediation path, not the final packed-index direct score-space scorer
- mixed-architecture Qwen stacks remain experimental as a whole
- linear-attention `ArraysCache` layers are pass-through, not TurboQuant-native
- throughput parity with `mlx_quant` is not a release target for this preview
- raw workspace archives are not release artifacts; use the exported preview bundle only

## Verification Authority

- unit and regression authority: `PYTHONPATH=src .venv/bin/python -m pytest -q`
- MLX smoke and benchmark authority: `.venv312` on Apple Silicon
- recommended smoke target:
  - `/Users/alican/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-4bit`

## Acceptance Snapshot

- `python3 -m compileall src` passed
- `PYTHONPATH=src .venv/bin/python -m pytest -q` -> `72 passed, 1 skipped`
- `turbomlx generate ... --scorer-mode native_mlx` passed

Benchmark snapshots on the tested Apple Silicon stack:

- `512/64`, warmup `1`, repeats `3`
  - `native_mlx`
    - `prompt_tps`: `380.04`
    - `generation_tps`: `42.71`
    - `key_path_bytes`: `28776896`
    - `total_kv_bytes`: `38198080`
    - `native_working_set_bytes`: `18841600`
  - `oracle_preview`
    - `prompt_tps`: `285.39`
    - `generation_tps`: `42.02`
    - `key_path_bytes`: `47618496`
    - `total_kv_bytes`: `57039680`
    - `native_working_set_bytes`: `0`
- `2048/64`, warmup `1`, repeats `3`
  - `native_mlx`
    - `prompt_tps`: `401.84`
    - `generation_tps`: `40.44`
    - `key_path_bytes`: `35264960`
    - `total_kv_bytes`: `69851968`
    - `native_working_set_bytes`: `69173248`
  - `oracle_preview`
    - `prompt_tps`: `285.52`
    - `generation_tps`: `39.50`
    - `key_path_bytes`: `104438208`
    - `total_kv_bytes`: `139025216`
    - `native_working_set_bytes`: `0`

Interpretation:

- this is a tested snapshot, not a throughput guarantee
- `native_mlx` is visibly active in the recorded runs because `scorer_route = native_mlx` and `native_working_set_bytes` is nonzero
- versus `oracle_preview`, `native_mlx` reduced key-path memory by `39.57%` at `512/64` and `66.23%` at `2048/64`
- versus `oracle_preview`, `native_mlx` reduced total KV bytes by `33.03%` at `512/64` and `49.76%` at `2048/64`
- versus `oracle_preview`, `native_mlx` improved prompt TPS by `33.16%` at `512/64` and `40.74%` at `2048/64`
- in this recorded snapshot, `native_mlx` also edged out `oracle_preview` on median decode TPS

## Distribution

- build the shareable artifact with `python3 scripts/export_preview_bundle.py`
- build a clean public source tree with `python3 scripts/export_preview_bundle.py --output-dir /ABS/PATH/Qwen3.5-TurboQuant-MLX-LM`
- distribute the resulting bundle under `dist/`, not a raw workspace zip
