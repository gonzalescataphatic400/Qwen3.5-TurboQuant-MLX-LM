#!/usr/bin/env python3
"""Create a clean shareable TurboMLX preview bundle or public source tree."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "turbomlx-preview-bundle.zip"
DEFAULT_PUBLIC_TREE = ROOT.parent / "Qwen3.5-TurboQuant-MLX-LM"
ALLOWLIST = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "docs",
    "scripts",
    "src",
    "tests",
)
EXCLUDED_NAMES = {
    ".pytest_cache",
    ".venv",
    ".venv312",
    "__MACOSX",
    "__pycache__",
    "build",
    "dist",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _should_include(relative_path: Path) -> bool:
    parts = set(relative_path.parts)
    if parts & EXCLUDED_NAMES:
        return False
    if relative_path.parts[:2] == ("docs", "superpowers"):
        return False
    if relative_path.name == ".DS_Store":
        return False
    if relative_path.suffix in EXCLUDED_SUFFIXES:
        return False
    if relative_path.parts[:1] == ("references",) and relative_path.suffix.lower() == ".pdf":
        return False
    return True


def iter_public_files():
    for item_name in ALLOWLIST:
        item_path = ROOT / item_name
        if not item_path.exists():
            continue
        if item_path.is_file():
            relative_path = item_path.relative_to(ROOT)
            if _should_include(relative_path):
                yield item_path
            continue
        for child in sorted(item_path.rglob("*")):
            if child.is_dir():
                continue
            relative_path = child.relative_to(ROOT)
            if _should_include(relative_path):
                yield child


def export_preview_bundle(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in iter_public_files():
            archive.write(file_path, arcname=file_path.relative_to(ROOT))
    return output_path


def export_public_tree(output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in iter_public_files():
        relative_path = file_path.relative_to(ROOT)
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)
    return output_dir


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination zip path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional destination directory for a clean public source tree export.",
    )
    parser.add_argument(
        "--export-public-tree",
        action="store_true",
        help=f"Export a clean public source tree to {DEFAULT_PUBLIC_TREE} unless --output-dir is provided.",
    )
    args = parser.parse_args()
    outputs: list[Path] = [export_preview_bundle(args.output)]
    if args.export_public_tree or args.output_dir is not None:
        public_destination = export_public_tree(args.output_dir or DEFAULT_PUBLIC_TREE)
        outputs.append(public_destination)
    for destination in outputs:
        print(destination)


if __name__ == "__main__":
    main()
