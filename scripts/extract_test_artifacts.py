#!/usr/bin/env python3
"""
Build and run the standalone C++ extractor for ONNX Runtime C++ test metadata.

The current extractor emits JSON rules for OpTester-based C++ tests, including
operator metadata and input tensors when they can be resolved statically from
the source. The JSON is intended to be reused by Python code later in the
pipeline to materialize ONNX models and optional output data.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parent.parent


def compile_cpp_extractor(extractor_source: Path, extractor_binary: Path, force: bool) -> None:
    """Build the standalone C++ extractor with g++ when needed."""
    if not force and extractor_binary.exists():
        if extractor_binary.stat().st_mtime >= extractor_source.stat().st_mtime:
            return

    extractor_binary.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-pedantic",
        str(extractor_source),
        "-o",
        str(extractor_binary),
    ]

    subprocess.run(command, check=True)


def run_cpp_extractor(
    extractor_binary: Path,
    source_path: Path,
    output_path: Path,
    all_domains: bool,
) -> None:
    """Execute the compiled extractor and write JSON output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(extractor_binary),
        "--source",
        str(source_path),
        "--output",
        str(output_path),
    ]

    if all_domains:
        command.append("--all-domains")

    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = repo_root()

    parser = argparse.ArgumentParser(
        description="Build and run the C++ ONNX Runtime test metadata extractor."
    )
    parser.add_argument(
        "--cpp-source",
        type=Path,
        default=root / "onnxruntime-org" / "onnxruntime" / "test" / "contrib_ops",
        help="C++ test file or directory to scan.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=root / "metadata" / "ort_cpp_test_rules.json",
        help="JSON file to write.",
    )
    parser.add_argument(
        "--extractor-source",
        type=Path,
        default=root / "cpp" / "ort_cpp_test_extractor.cpp",
        help="Path to the C++ extractor source file.",
    )
    parser.add_argument(
        "--extractor-binary",
        type=Path,
        default=root / "build" / "ort_cpp_test_extractor.exe",
        help="Path to the compiled extractor binary.",
    )
    parser.add_argument(
        "--all-domains",
        action="store_true",
        help="Include non-kMSDomain tests in the output.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a rebuild of the extractor binary.",
    )
    return parser.parse_args()


def main() -> int:
    """Build and run the extractor."""
    args = parse_args()

    try:
        compile_cpp_extractor(args.extractor_source, args.extractor_binary, args.rebuild)
        run_cpp_extractor(
            args.extractor_binary,
            args.cpp_source,
            args.json_output,
            args.all_domains,
        )
    except subprocess.CalledProcessError as error:
        print(f"Extractor command failed with exit code {error.returncode}.", file=sys.stderr)
        return error.returncode

    print(f"JSON written to {args.json_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
