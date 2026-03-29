# Qwen3.5-TurboQuant-MLX-LM

`TurboMLX v0.1 Research Preview`

This repository packages the TurboMLX preview work for GitHub under the name `Qwen3.5-TurboQuant-MLX-LM`. The Python package and CLI remain `turbomlx`.

TurboMLX `v0.1 Research Preview` currently targets Qwen3 / Qwen3.5 full-attention `KVCache` layers only.

Its public contract for the current preview is:

- paper-faithful key-path implementations of `TurboQuantmse` and `TurboQuantprod`
- TurboMLX-owned prompt-cache save/load wrappers for a TurboQuant KV backend
- reference math utilities, mixed-precision paper profiles, and eval helpers

Important limitations:

- values are dense by default
- end-to-end KV behavior is therefore not fully paper-equivalent unless value quantization is enabled
- runtime preview is Qwen-first and currently patches `qwen3_next` plus the shared `mlx_lm.models.base` dispatch symbol
- mixed-architecture Qwen stacks remain experimental as a whole; TurboQuant conversion applies only to full-attention `KVCache` layers and leaves linear-attention `ArraysCache` layers untouched
- rotating/sliding-window families remain unsupported in preview
- `v0.1 Research Preview` focuses on correctness and quality gates, not throughput leadership
- preview runtime scoring defaults to `oracle_preview`; a narrow `native_mlx` scorer preview now exists only for Qwen3 / Qwen3.5 full-attention `KVCache` with `mode=mse`, `bits_total=4`, and `values_mode=dense`
- `native_mlx` is a Stage A remediation path, not the final packed-index direct score-space scorer
- the supported public runtime entrypoints are `generate_with_backend`, `convert_prompt_cache`, `save_prompt_cache`, and `load_prompt_cache`

## Release Status

- release label: `v0.1 Research Preview`
- package identity: `turbomlx`
- CLI: `turbomlx`
- supported public preview target: Qwen3 / Qwen3.5 full-attention `KVCache` only
- non-goal for this release: throughput parity with `mlx_quant`

## Latest Verification Snapshot

Tested on `2026-03-29` with:

- `mlx==0.31.1`
- `mlx-lm==0.31.1`
- smoke model: `mlx-community/Qwen3.5-9B-MLX-4bit`

Verification results:

- `python3 -m compileall src` passed
- `PYTHONPATH=src .venv/bin/python -m pytest -q` -> `72 passed, 1 skipped`
- Qwen native smoke generate passed with `scorer_route = native_mlx`

Illustrative benchmark snapshot on the tested Apple Silicon stack:

| Route | Prompt TPS | Decode TPS | Key Path Bytes | Native Working Set Bytes |
| --- | ---: | ---: | ---: | ---: |
| `native_mlx` median (`512/64`, warmup `1`, repeats `3`) | `381.41` | `40.10` | `28776896` | `18841600` |
| `oracle_preview` median (`512/64`, warmup `1`, repeats `3`) | `285.61` | `41.58` | `47618496` | `0` |

Interpretation:

- this snapshot is environment-specific and not a throughput guarantee
- in this run, `native_mlx` improved prompt throughput and reduced key-path bytes materially
- in this same run, `oracle_preview` was still slightly faster on median decode TPS

## Bit Semantics

- `bits_total` is the user-facing total per-channel bit budget
- `mode=mse`: all `bits_total` go to the Lloyd-Max main quantizer
- `mode=prod`: `bits_mse = bits_total - 1`, plus a 1-bit QJL residual
- `mode=prod` is supported only for `bits_total >= 2`
- default policy:
  - `1-bit`: `mse`
  - `2-bit`: `prod`
  - `3/4-bit`: `mse`

## Mixed Precision Paper Profile

Paper-style non-integer effective settings such as `2.5` and `3.5` bits are
represented as explicit mixed-precision outlier profiles.

Supported profile knobs:

- `outlier_channels`
- `outlier_high_bits`
- `regular_bits`
- `outlier_selection_policy`

If mixed precision is disabled, all quality and memory claims are restricted to
integer-bit configurations.

## Release Policy

- `v0.1 Research Preview`
  - correctness
  - serialization stability
  - prompt-cache continuity
  - long-context quality helpers
  - honest benchmark reporting
- `v1.0 Stable`
  - all of the above
  - explicit `mlx_quant` decode parity target on the reference benchmark matrix

## Scope

Official preview target:

- Qwen3 / Qwen3.5 full-attention layers that use the default non-rotating `KVCache` path

Experimental:

- mixed full-attention / linear-attention Qwen stacks as a whole
- value quantization on the dense-key preview fallback path
- rotating or sliding-window cache families

## Preview Eval Surface

- `eval-needle` is a preview retrieval harness that inserts the needle only inside the haystack context
- `eval-jsonl` is a local JSONL exact/substring harness and is not a real LongBench-E implementation
- prompt-cache roundtrip support is provided by `turbomlx.save_prompt_cache()` and `turbomlx.load_prompt_cache()`, not by upstream `mlx-lm` loaders

## Prompt-Cache Policy

- prompt-cache files are trusted-local-only and currently remain `pickle`-backed
- schema `v2` is the current write format and includes `cache_type_id` metadata
- schema `v1` files still load in read-only compatibility mode via deprecated `class_path` fallback
- if you load an older cache, re-save it to migrate to `v2`

## Qwen Preview Runtime

- Qwen3 / Qwen3.5 preview correctness uses a dense reconstructed key fallback on full-attention layers
- grouped-query attention math is delegated back to MLX native SDPA after reconstructing dense keys from the TurboQuant cache
- this path is correctness-first and intentionally preview-grade; it is not a throughput claim
- `--scorer-mode native_mlx` is a narrower preview path layered on top of the same Qwen-first contract
- supported `native_mlx` config:
  - Qwen3 / Qwen3.5
  - full-attention `KVCache`
  - `mode=mse`
  - `bits_total=4`
  - `values_mode=dense`
- unsupported `native_mlx` combinations emit a warning once per reason and fall back to the preview scorer path
- the current native scorer is still an intermediate on-device remediation step, not the final packed-index direct-score architecture

## Verification

- unit and regression suite: `PYTHONPATH=src .venv/bin/python -m pytest -q`
- MLX smoke and benchmark authority: `.venv312`
- recommended smoke target: `TURBOMLX_SMOKE_QWEN_MODEL=/Users/alican/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-4bit`
- benchmark methodology for current preview work:
  - use at least 1 warmup run
  - use at least 3 measured repeats
  - report median results
  - inspect `scorer_route` in the output to verify whether `native_mlx` actually ran or fell back
  - inspect `timed_generation_tokens` and `native_working_set_bytes` in the output when comparing preview scorer routes

## Preview Bundle

- this source tree is a preview-candidate working tree, not a release-ready artifact
- use `scripts/export_preview_bundle.py` to create a clean shareable preview bundle under `dist/`
- use `python3 scripts/export_preview_bundle.py --output-dir /ABS/PATH/Qwen3.5-TurboQuant-MLX-LM` to create a clean public source tree
- distribute the exported preview bundle, not a raw workspace zip
- the exported bundle excludes local virtualenvs, cache directories, transient artifacts, and reference PDFs

## Tested Runtime Stack

- `mlx==0.31.1`
- `mlx-lm==0.31.1`

## Dates

- arXiv v1: `28 Apr 2025`
- OpenReview / ICLR 2026 Poster entry: `26 Jan 2026`

This repository started from a blank directory plus the TurboQuant paper, so
the current implementation emphasizes clean interfaces and verifiable reference
math first. MLX runtime hardening is intentionally staged behind the preview
release boundary.
