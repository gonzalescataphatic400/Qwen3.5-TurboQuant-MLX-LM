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

## Distribution

- build the shareable artifact with `python3 scripts/export_preview_bundle.py`
- build a clean public source tree with `python3 scripts/export_preview_bundle.py --output-dir /ABS/PATH/Qwen3.5-TurboQuant-MLX-LM`
- distribute the resulting bundle under `dist/`, not a raw workspace zip
