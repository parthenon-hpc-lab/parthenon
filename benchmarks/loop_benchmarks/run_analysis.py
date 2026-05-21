#!/usr/bin/env python3

import argparse
import csv
import math
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "parthenon-loop-mpl")
)
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

VARIANT_STYLE = {
    "kokkos_dense_flat_bvkji": ("#1d3557", "s"),
    "kokkos_flat_kji": ("#264653", "o"),
    "kokkos_mdrange_kji": ("#287271", "D"),
    "gpu_rawspan_chunk_ninj": ("#6d597a", "^"),
    "gpu_rawspan_chunk_ni": ("#8ab17d", "P"),
    "gpu_rawspan_sweep": ("#8ab17d", "P"),
    "gpu_rawspan_view_chunk_ninj": ("#457b9d", ">"),
    "gpu_rawspan_view_chunk_ni": ("#a8dadc", "<"),
    "gpu_rawspan_view_sweep": ("#a8dadc", "<"),
    "gpu_logical_chunk_ninj": ("#bc6c25", "v"),
    "gpu_logical_chunk_ni": ("#dda15e", "X"),
    "gpu_logical_sweep": ("#dda15e", "X"),
    "cpu_dense_flat_bvkji": ("#7f5539", "s"),
    "cpu_logical_kji": ("#0f4c5c", "o"),
    "cpu_rawspan_voi": ("#2a9d8f", "P"),
    "cpu_rawspan_ovi": ("#b56576", "s"),
    "kokkos_rawspan_ovi": ("#6d597a", "^"),
    "kokkos_rawspan_view_ovi": ("#457b9d", ">"),
    "cpu_logical_ovi": ("#bc6c25", "v"),
    "kokkos_logical_ovi": ("#dda15e", "X"),
}

CPU_BASE_VARIANTS = [
    "cpu_dense_flat_bvkji",
    "cpu_logical_kji",
    "cpu_rawspan_voi",
    "cpu_rawspan_ovi",
    "cpu_logical_ovi",
    "kokkos_rawspan_ovi",
    "kokkos_rawspan_view_ovi",
    "kokkos_logical_ovi",
]

CPU_CHUNK_SWEEP_VARIANTS = [
    "cpu_rawspan_voi",
    "cpu_rawspan_ovi",
    "cpu_logical_ovi",
    "kokkos_rawspan_ovi",
    "kokkos_logical_ovi",
]

CPU_CHUNK_SWEEP_BASELINES = ["cpu_logical_kji"]

GPU_BASE_VARIANTS = [
    "kokkos_dense_flat_bvkji",
    "kokkos_flat_kji",
    "kokkos_mdrange_kji",
    "gpu_rawspan_chunk_ninj",
    "gpu_rawspan_chunk_ni",
    "gpu_rawspan_view_chunk_ninj",
    "gpu_rawspan_view_chunk_ni",
    "gpu_logical_chunk_ninj",
    "gpu_logical_chunk_ni",
]

GPU_CHUNK_SWEEP_VARIANTS = [
    "gpu_rawspan_sweep",
    "gpu_rawspan_view_sweep",
    "gpu_logical_sweep",
]

GPU_CHUNK_SWEEP_BASELINES = ["kokkos_flat_kji", "kokkos_mdrange_kji"]


class ProgressReporter:
    def __init__(self, total_runs):
        self.total_runs = total_runs
        self.completed_runs = 0

    def log_completed(self, experiment, row):
        self.completed_runs += 1
        details = [
            f"kernel={row['kernel']}",
            f"variant={row['variant']}",
            f"ni={row['ni']}",
        ]
        if "chunk_sweep" in experiment:
            details.append(f"chunk={row['inner_chunk_length']}")
        details.append(f"blocks={row['blocks']}")
        details.append(f"min_seconds={row['min_seconds']:.6f}")
        details.append(f"updates_per_second={row['updates_per_second']:.3e}")
        print(
            f"[{self.completed_runs}/{self.total_runs}] {experiment}: "
            + ", ".join(details),
            flush=True,
        )


def variant_labels(analysis_mode, run_set):
    raw_chunk_label = "ninner=ni" if analysis_mode == "verify" else "ninner=ni*nj"
    cpu_labels = {
        "cpu_dense_flat_bvkji": "CPU dense full-memory (b, v, k, j, i)",
        "cpu_logical_kji": "CPU logical (v, kji)",
        "cpu_rawspan_voi": f"CPU raw-memory (v, outer, inner), {raw_chunk_label}",
        "cpu_rawspan_ovi": f"CPU raw-memory (outer, v, inner), {raw_chunk_label}",
        "cpu_logical_ovi": f"CPU logical (outer, v, inner), {raw_chunk_label}",
        "kokkos_rawspan_ovi": f"Kokkos raw-memory (outer, v, inner), {raw_chunk_label}",
        "kokkos_rawspan_view_ovi": f"Kokkos raw-memory + View indexing (outer, v, inner), {raw_chunk_label}",
        "kokkos_logical_ovi": f"Kokkos logical (outer, v, inner), {raw_chunk_label}",
    }
    if run_set == "gpu":
        return {
            "kokkos_dense_flat_bvkji": "Kokkos dense full-memory (b, v, k, j, i)",
            "kokkos_flat_kji": "Kokkos flat logical (kji)",
            "kokkos_mdrange_kji": "Kokkos MDRange logical (kji)",
            "gpu_rawspan_chunk_ninj": "Kokkos raw-memory (outer, v, inner), ninner=ni*nj",
            "gpu_rawspan_chunk_ni": "Kokkos raw-memory (outer, v, inner), ninner=ni",
            "gpu_rawspan_view_chunk_ninj": "Kokkos raw-memory + View indexing, ninner=ni*nj",
            "gpu_rawspan_view_chunk_ni": "Kokkos raw-memory + View indexing, ninner=ni",
            "gpu_rawspan_sweep": "Kokkos raw-memory (outer, v, inner)",
            "gpu_rawspan_view_sweep": "Kokkos raw-memory + View indexing",
            "gpu_logical_chunk_ninj": "Kokkos logical (outer, v, inner), ninner=ni*nj",
            "gpu_logical_chunk_ni": "Kokkos logical (outer, v, inner), ninner=ni",
            "gpu_logical_sweep": "Kokkos logical (outer, v, inner)",
        }
    return cpu_labels


def chunk_sweep_variant_labels(run_set):
    cpu_labels = {
        "cpu_logical_kji": "CPU logical (v, kji)",
        "cpu_rawspan_voi": "CPU raw-memory (v, outer, inner)",
        "cpu_rawspan_ovi": "CPU raw-memory (outer, v, inner)",
        "cpu_logical_ovi": "CPU logical (outer, v, inner)",
        "kokkos_rawspan_ovi": "Kokkos raw-memory (outer, v, inner)",
        "kokkos_logical_ovi": "Kokkos logical (outer, v, inner)",
    }
    if run_set == "gpu":
        return {
            "kokkos_dense_flat_bvkji": "Kokkos dense full-memory (b, v, k, j, i)",
            "kokkos_flat_kji": "Kokkos flat logical (kji)",
            "kokkos_mdrange_kji": "Kokkos MDRange logical (kji)",
            "gpu_rawspan_sweep": "Kokkos raw-memory (outer, v, inner)",
            "gpu_rawspan_view_sweep": "Kokkos raw-memory + View indexing",
            "gpu_logical_sweep": "Kokkos logical (outer, v, inner)",
        }
    return cpu_labels


def experiment_labels(experiment, analysis_mode, run_set):
    if experiment.endswith("chunk_sweep"):
        return chunk_sweep_variant_labels(run_set)
    return variant_labels(analysis_mode, run_set)


def ordered_experiment_labels(rows, experiment, labels):
    seen = set()
    ordered = []
    for row in rows:
        if row["experiment"] != experiment:
            continue
        label = labels.get(row["variant"], row["variant"])
        if label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered


def verify_chunk_mode(analysis_mode):
    return "row" if analysis_mode == "verify" else None


def chunk_flag_text(variant, analysis_mode):
    if variant in {"cpu_dense_flat_bvkji", "kokkos_dense_flat_bvkji"}:
        return "(dense full-memory baseline)"
    if variant in {"cpu_rawspan_voi", "cpu_rawspan_ovi", "cpu_logical_ovi"}:
        return (
            "--inner-chunk-length <ni>"
            if analysis_mode == "verify"
            else "(default ninner=ni*nj)"
        )
    if variant in {
        "kokkos_rawspan_ovi",
        "kokkos_rawspan_view_ovi",
        "kokkos_logical_ovi",
    }:
        return (
            "--inner-chunk-length <ni>"
            if analysis_mode == "verify"
            else "(default ninner=ni*nj)"
        )
    if variant == "gpu_rawspan_chunk_ninj":
        return "--inner-chunk-length <ni*nj>"
    if variant == "gpu_rawspan_chunk_ni":
        return "--inner-chunk-length <ni>"
    if variant == "gpu_rawspan_sweep":
        return "--inner-chunk-length <chunk>"
    if variant == "gpu_rawspan_view_chunk_ninj":
        return "--inner-chunk-length <ni*nj>"
    if variant == "gpu_rawspan_view_chunk_ni":
        return "--inner-chunk-length <ni>"
    if variant == "gpu_rawspan_view_sweep":
        return "--inner-chunk-length <chunk>"
    if variant == "gpu_logical_chunk_ninj":
        return "--inner-chunk-length <ni*nj>"
    if variant == "gpu_logical_chunk_ni":
        return "--inner-chunk-length <ni>"
    if variant == "gpu_logical_sweep":
        return "--inner-chunk-length <chunk>"
    return ""


def default_chunk_description(args):
    return (
        "default inner chunk length = ni*nj "
        f"(so at fixed chunk-sweep ni=nj=nk={args.chunk_sweep_ni}, default={args.chunk_sweep_ni ** 2})"
    )


def analysis_variants(run_set):
    if run_set == "gpu":
        return GPU_BASE_VARIANTS, GPU_CHUNK_SWEEP_VARIANTS, GPU_CHUNK_SWEEP_BASELINES
    return CPU_BASE_VARIANTS, CPU_CHUNK_SWEEP_VARIANTS, CPU_CHUNK_SWEEP_BASELINES


def parse_count(value):
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid count value: {value}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"count must be positive: {value}")
    rounded = int(parsed)
    if rounded <= 0:
        raise argparse.ArgumentTypeError(f"count must be positive: {value}")
    return rounded


def parse_csv_ints(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the standard Parthenon loop benchmark analysis suite."
    )
    parser.add_argument(
        "--binary",
        default="build/benchmarks/loop_benchmarks/loop-benchmarks",
        help="Path to the benchmark executable",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/analysis",
        help="Directory for generated CSV and PDF outputs",
    )
    parser.add_argument("--vars", type=int, default=16, help="Number of variables")
    parser.add_argument("--ghosts", type=int, default=2, help="Ghost zone width")
    parser.add_argument(
        "--edge-values",
        dest="edge_values",
        default="8,16,32,64,128",
        help="Comma-separated cubic block-edge lengths to sweep (ni=nj=nk=edge)",
    )
    parser.add_argument(
        "--ni-values",
        dest="edge_values",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--target-total-cells",
        type=parse_count,
        default=None,
        help="Choose blocks so blocks * ni * nj * nk stays close to this value; accepts forms like 1e9",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Repeats per run")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument(
        "--heavy-iterations",
        type=int,
        default=8,
        help="Heavy-kernel iteration count for the heavy sweep",
    )
    parser.add_argument(
        "--heavy-iteration-values",
        default="1,2,4,8,16,32,64",
        help="Comma-separated heavy-iteration counts for the fixed-size intensity sweep",
    )
    parser.add_argument(
        "--skip-heavy-intensity-sweep",
        action="store_true",
        help="Skip the fixed-size heavy-iteration intensity sweep",
    )
    parser.add_argument(
        "--title",
        default="Parthenon Loop Benchmark Analysis",
        help="PDF title",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=("default", "verify"),
        default="default",
        help="Use default chunking for report mode or force CPU raw-span chunk=ni for verification mode",
    )
    parser.add_argument(
        "--chunk-sweep-ni",
        type=int,
        default=32,
        help="Fixed ni value used for the chunk-size sweep",
    )
    parser.add_argument(
        "--chunk-values",
        default="8,16,32,64,128,256,512,1024",
        help="Comma-separated inner chunk lengths to sweep at fixed cubic edge length",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run only GPU-capable Kokkos variants instead of the default CPU-focused suite",
    )
    return parser.parse_args()


def run_command(cmd):
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return completed.stdout


def try_run_command(cmd):
    try:
        return run_command(cmd).strip()
    except Exception:
        return ""


def parse_summary(stdout_text):
    row = {}
    for line in stdout_text.strip().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        row[key.strip()] = value.strip()

    shape = row.pop("shape", "")
    if shape:
        for part in shape.split():
            name, value = part.split(":", 1)
            if name == "vars":
                name = "variables"
            row[name] = value

    active_range = row.pop("active_range", "")
    if active_range:
        active_min, active_max = active_range.split("..", 1)
        row["active_min"] = active_min
        row["active_max"] = active_max
    return row


def numericize(row):
    result = dict(row)
    result["ragged"] = result["ragged"].lower() == "true"
    int_fields = {
        "blocks",
        "variables",
        "nk",
        "nj",
        "ni",
        "ghost_zones",
        "active_min",
        "active_max",
        "inner_chunk_length",
        "explicit_team_size",
        "heavy_iterations",
        "repeats",
        "total_updates",
    }
    float_fields = {
        "min_seconds",
        "median_seconds",
        "mean_seconds",
        "updates_per_second",
        "estimated_bandwidth_gb_s",
        "estimated_flops_per_update",
        "arithmetic_intensity_flops_per_byte",
    }
    for field in int_fields:
        result[field] = int(float(result[field]))
    for field in float_fields:
        result[field] = float(result[field])
    return result


def detect_cpu_description():
    candidates = []
    system = platform.system()
    if system == "Darwin":
        hardware = try_run_command(["system_profiler", "SPHardwareDataType"])
        if hardware:
            chip = ""
            model_name = ""
            for line in hardware.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == "Chip" and value:
                    chip = value
                elif key == "Model Name" and value:
                    model_name = value
            if chip and model_name:
                candidates.append(f"{model_name} ({chip})")
            elif chip:
                candidates.append(chip)
            elif model_name:
                candidates.append(model_name)
        candidates.extend(
            [
                try_run_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                try_run_command(["sysctl", "-n", "hw.model"]),
            ]
        )
    elif system == "Linux":
        lscpu = try_run_command(["lscpu"])
        if lscpu:
            for line in lscpu.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key.strip() in {"Model name", "Hardware"} and value.strip():
                    candidates.append(value.strip())
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key.strip() in {"model name", "Hardware"} and value.strip():
                    candidates.append(value.strip())
                    break

    processor = platform.processor().strip()
    if processor:
        candidates.append(processor)

    machine = platform.machine().strip()
    if machine:
        candidates.append(machine)

    for candidate in candidates:
        if candidate:
            return candidate
    return "unknown"


def find_build_dir(binary):
    binary_path = Path(binary).resolve()
    for parent in binary_path.parents:
        if (parent / "CMakeCache.txt").exists():
            return parent
    return None


def parse_cmake_cache(cache_path):
    cache = {}
    if not cache_path.exists():
        return cache
    for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if (
            not line
            or line.startswith(("#", "//"))
            or "=" not in line
            or ":" not in line
        ):
            continue
        key_type, value = line.split("=", 1)
        key, _sep, _value_type = key_type.partition(":")
        cache[key] = value
    return cache


def detect_execution_target_from_cache(cache):
    device_keys = [
        "Kokkos_ENABLE_CUDA",
        "Kokkos_ENABLE_HIP",
        "Kokkos_ENABLE_SYCL",
        "Kokkos_ENABLE_OPENMPTARGET",
        "Kokkos_ENABLE_OPENACC",
    ]
    for key in device_keys:
        if cache.get(key, "").upper() == "ON":
            return "device"
    return "host"


def parse_flags_make(flags_make_path):
    info = {}
    if not flags_make_path.exists():
        return info
    for line in flags_make_path.read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        info[key.strip()] = value.strip()
    return info


def collect_system_info(binary):
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": detect_cpu_description(),
    }
    info["uname"] = try_run_command(["uname", "-a"]) or "unknown"

    build_dir = find_build_dir(binary)
    if build_dir is None:
        return info

    info["build_dir"] = str(build_dir)
    cache = parse_cmake_cache(build_dir / "CMakeCache.txt")
    info["execution_target"] = detect_execution_target_from_cache(cache)
    info["build_type"] = cache.get("CMAKE_BUILD_TYPE", "unknown")
    info["compiler"] = cache.get("CMAKE_CXX_COMPILER", "unknown")

    compiler = info["compiler"]
    if compiler and compiler != "unknown":
        compiler_version = try_run_command([compiler, "--version"])
        if compiler_version:
            info["compiler_version"] = compiler_version.splitlines()[0]

    build_type = info["build_type"]
    common_flags = cache.get("CMAKE_CXX_FLAGS", "").strip()
    build_flags = cache.get(f"CMAKE_CXX_FLAGS_{build_type.upper()}", "").strip()
    info["cmake_cxx_flags"] = " ".join(
        part for part in [common_flags, build_flags] if part
    )

    flags_make = (
        build_dir
        / "benchmarks/loop_benchmarks/CMakeFiles/loop-benchmarks.dir/flags.make"
    )
    target_flags = parse_flags_make(flags_make)
    info["target_compile_flags"] = target_flags.get("CXX_FLAGS", "")
    info["target_defines"] = target_flags.get("CXX_DEFINES", "")
    return info


def default_target_total_cells(binary):
    build_dir = find_build_dir(binary)
    if build_dir is None:
        return 2_000_000
    cache = parse_cmake_cache(build_dir / "CMakeCache.txt")
    execution_target = detect_execution_target_from_cache(cache)
    if execution_target == "device":
        return 20_000_000
    return 2_000_000


def append_wrapped_line(lines, label, value, width=100):
    if not value:
        return
    wrapped = textwrap.wrap(
        f"{label}{value}",
        width=width,
        subsequent_indent=" " * len(label),
        break_long_words=False,
        break_on_hyphens=False,
    )
    lines.extend(wrapped or [f"{label}{value}"])


def blocks_for_shape(edge, args):
    cells_per_block = edge * edge * edge
    return max(1, int(round(args.target_total_cells / cells_per_block)))


def build_benchmark_command(
    binary, csv_path, kernel, variant, edge, args, chunk_length=None
):
    cmd = [
        binary,
        "--kernel",
        kernel,
        "--variant",
        variant,
        "--blocks",
        str(blocks_for_shape(edge, args)),
        "--vars",
        str(args.vars),
        "--nk",
        str(edge),
        "--nj",
        str(edge),
        "--ni",
        str(edge),
        "--ghosts",
        str(args.ghosts),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--csv",
        str(csv_path),
    ]
    if chunk_length is not None:
        cmd.extend(["--inner-chunk-length", str(chunk_length)])
    if kernel == "heavy":
        cmd.extend(["--heavy-iterations", str(args.heavy_iterations)])
    return cmd


def gpu_variant_command_settings(variant, edge, args, chunk_length=None):
    if variant == "gpu_rawspan_chunk_ninj":
        return "kokkos_rawspan_ovi", edge * edge
    if variant == "gpu_rawspan_chunk_ni":
        return "kokkos_rawspan_ovi", edge
    if variant == "gpu_rawspan_sweep":
        return "kokkos_rawspan_ovi", chunk_length
    if variant == "gpu_rawspan_view_chunk_ninj":
        return "kokkos_rawspan_view_ovi", edge * edge
    if variant == "gpu_rawspan_view_chunk_ni":
        return "kokkos_rawspan_view_ovi", edge
    if variant == "gpu_rawspan_view_sweep":
        return "kokkos_rawspan_view_ovi", chunk_length
    if variant == "gpu_logical_chunk_ninj":
        return "kokkos_logical_ovi", edge * edge
    if variant == "gpu_logical_chunk_ni":
        return "kokkos_logical_ovi", edge
    if variant == "gpu_logical_sweep":
        return "kokkos_logical_ovi", chunk_length
    return variant, chunk_length


def run_sweep(
    binary, csv_path, kernel, variants, edge_values, args, analysis_mode, progress
):
    rows = []
    chunk_mode = verify_chunk_mode(analysis_mode)
    for edge in edge_values:
        for variant in variants:
            chunk_length = None
            if (
                variant
                in {
                    "cpu_rawspan_voi",
                    "cpu_rawspan_ovi",
                    "cpu_logical_ovi",
                    "kokkos_rawspan_ovi",
                    "kokkos_rawspan_view_ovi",
                    "kokkos_logical_ovi",
                }
                and chunk_mode == "row"
            ):
                chunk_length = edge
            cmd_variant = variant
            if args.gpu:
                cmd_variant, chunk_length = gpu_variant_command_settings(
                    variant, edge, args, chunk_length
                )
            cmd = build_benchmark_command(
                binary, csv_path, kernel, cmd_variant, edge, args, chunk_length
            )
            summary = parse_summary(run_command(cmd))
            summary["variant"] = variant
            summary["repeats"] = str(args.repeats)
            summary["experiment"] = f"{kernel}_ni_sweep"
            row = numericize(summary)
            rows.append(row)
            progress.log_completed(summary["experiment"], row)
    return rows


def run_chunk_sweep(
    binary, csv_path, kernel, variants, baseline_variants, chunk_values, args, progress
):
    rows = []
    ni = args.chunk_sweep_ni
    for chunk_length in chunk_values:
        for variant in variants:
            cmd_variant = variant
            if args.gpu:
                cmd_variant, chunk_length = gpu_variant_command_settings(
                    variant, ni, args, chunk_length
                )
            cmd = build_benchmark_command(
                binary, csv_path, kernel, cmd_variant, ni, args, chunk_length
            )
            summary = parse_summary(run_command(cmd))
            summary["variant"] = variant
            summary["repeats"] = str(args.repeats)
            summary["experiment"] = f"{kernel}_chunk_sweep"
            row = numericize(summary)
            rows.append(row)
            progress.log_completed(summary["experiment"], row)
    for variant in baseline_variants:
        cmd = build_benchmark_command(binary, csv_path, kernel, variant, ni, args)
        summary = parse_summary(run_command(cmd))
        summary["variant"] = variant
        summary["repeats"] = str(args.repeats)
        summary["experiment"] = f"{kernel}_chunk_sweep"
        row = numericize(summary)
        rows.append(row)
        progress.log_completed(summary["experiment"], row)
    return rows


def run_heavy_intensity_sweep(
    binary, csv_path, variants, edge, heavy_iteration_values, args, progress
):
    rows = []
    chunk_mode = verify_chunk_mode(args.analysis_mode)
    for heavy_iterations in heavy_iteration_values:
        for variant in variants:
            chunk_length = None
            if (
                variant
                in {
                    "cpu_rawspan_voi",
                    "cpu_rawspan_ovi",
                    "cpu_logical_ovi",
                    "kokkos_rawspan_ovi",
                    "kokkos_rawspan_view_ovi",
                    "kokkos_logical_ovi",
                }
                and chunk_mode == "row"
            ):
                chunk_length = edge
            cmd_variant = variant
            if args.gpu:
                cmd_variant, chunk_length = gpu_variant_command_settings(
                    variant, edge, args, chunk_length
                )
            cmd = build_benchmark_command(
                binary, csv_path, "heavy", cmd_variant, edge, args, chunk_length
            )
            cmd.extend(["--heavy-iterations", str(heavy_iterations)])
            summary = parse_summary(run_command(cmd))
            summary["variant"] = variant
            summary["repeats"] = str(args.repeats)
            summary["heavy_iterations"] = str(heavy_iterations)
            summary["experiment"] = "heavy_intensity_sweep"
            row = numericize(summary)
            rows.append(row)
            progress.log_completed(summary["experiment"], row)
    return rows


def count_total_runs(
    base_variants,
    chunk_sweep_variants,
    chunk_sweep_baselines,
    ni_values,
    chunk_values,
    heavy_iteration_values,
    include_heavy_intensity_sweep,
):
    return (
        2 * len(base_variants) * len(ni_values)
        + 2 * len(chunk_sweep_variants) * len(chunk_values)
        + 2 * len(chunk_sweep_baselines)
        + (
            len(base_variants) * len(heavy_iteration_values)
            if include_heavy_intensity_sweep
            else 0
        )
    )


def write_combined_csv(path, rows):
    fieldnames = [
        "experiment",
        "backend",
        "variant",
        "kernel",
        "ragged",
        "blocks",
        "variables",
        "nk",
        "nj",
        "ni",
        "ghost_zones",
        "active_min",
        "active_max",
        "inner_chunk_length",
        "team_size_mode",
        "explicit_team_size",
        "heavy_iterations",
        "repeats",
        "min_seconds",
        "median_seconds",
        "mean_seconds",
        "updates_per_second",
        "estimated_bandwidth_gb_s",
        "estimated_flops_per_update",
        "arithmetic_intensity_flops_per_byte",
        "total_updates",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_title_page(
    pdf, title, args, system_info, rows, labels, chunk_values, heavy_iteration_values
):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ni_values = sorted({row["ni"] for row in rows})
    lines = [
        title,
        "",
        f"Binary: {args.binary}",
        f"variant set: {'gpu' if args.gpu else 'cpu'}",
        f"analysis mode: {args.analysis_mode}",
        f"edge sweep (ni=nj=nk): {ni_values}",
        f"chunk sweep: ni=nj=nk={args.chunk_sweep_ni}, chunks={chunk_values}",
        (
            f"heavy intensity sweep: ni=nj=nk={args.chunk_sweep_ni}, heavy_iterations={heavy_iteration_values}"
            if not args.skip_heavy_intensity_sweep
            else "heavy intensity sweep: skipped"
        ),
        f"shape base: vars={args.vars}, ghosts={args.ghosts}",
        default_chunk_description(args),
        f"target total cells: {args.target_total_cells}",
        f"repeats={args.repeats}, warmup={args.warmup}, heavy_iterations={args.heavy_iterations}",
        "",
        f"Platform: {system_info.get('platform', 'unknown')}",
        f"CPU: {system_info.get('cpu', 'unknown')}",
        f"uname: {system_info.get('uname', 'unknown')}",
    ]
    append_wrapped_line(lines, "Build dir: ", system_info.get("build_dir", ""))
    append_wrapped_line(lines, "Build type: ", system_info.get("build_type", ""))
    append_wrapped_line(lines, "Compiler: ", system_info.get("compiler", ""))
    append_wrapped_line(
        lines, "Compiler version: ", system_info.get("compiler_version", "")
    )
    append_wrapped_line(
        lines, "Target compile flags: ", system_info.get("target_compile_flags", "")
    )
    append_wrapped_line(
        lines, "Target defines: ", system_info.get("target_defines", "")
    )
    append_wrapped_line(
        lines, "CMake CXX flags: ", system_info.get("cmake_cxx_flags", "")
    )
    lines.extend(
        [
            "",
            "Experiments:",
            (
                (
                    "- stencil edge sweep: "
                    + " vs ".join(
                        ordered_experiment_labels(
                            rows,
                            "stencil_ni_sweep",
                            experiment_labels(
                                "stencil_ni_sweep",
                                args.analysis_mode,
                                "gpu" if args.gpu else "cpu",
                            ),
                        )
                    )
                )
                if ni_values
                else ""
            ),
            (
                (
                    "- heavy edge sweep: "
                    + " vs ".join(
                        ordered_experiment_labels(
                            rows,
                            "heavy_ni_sweep",
                            experiment_labels(
                                "heavy_ni_sweep",
                                args.analysis_mode,
                                "gpu" if args.gpu else "cpu",
                            ),
                        )
                    )
                )
                if ni_values
                else ""
            ),
            (
                f"- chunk sweeps at ni=nj=nk={args.chunk_sweep_ni}: raw-span variants and "
                "selected baselines across explicit inner chunk lengths"
            ),
            (
                (
                    f"- heavy intensity sweep at ni=nj=nk={args.chunk_sweep_ni}: "
                    "same variants with varying heavy-iteration count"
                )
                if not args.skip_heavy_intensity_sweep
                else "- heavy intensity sweep skipped"
            ),
        ]
    )
    ax.text(
        0.05,
        0.97,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(ax, rows, experiment, title, x_key, x_label, labels):
    exp_rows = [row for row in rows if row["experiment"] == experiment]
    by_variant = defaultdict(list)
    for row in exp_rows:
        by_variant[row["variant"]].append((row[x_key], row["updates_per_second"]))

    for variant, samples in sorted(by_variant.items()):
        samples.sort()
        xs = [x for x, _ in samples]
        ups = [ups for _, ups in samples]
        color, marker = VARIANT_STYLE.get(variant, ("#333333", "o"))
        ax.plot(xs, ups, marker=marker, color=color, label=labels.get(variant, variant))

    ax.set_xscale("log", base=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("updates/s")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()


def add_sweep_pages(pdf, rows, analysis_mode, run_set):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plot_sweep(
        axes[0],
        rows,
        "stencil_ni_sweep",
        "Stencil Cubic Edge Sweep",
        "ni",
        "edge length n (ni=nj=nk=n)",
        experiment_labels("stencil_ni_sweep", analysis_mode, run_set),
    )
    plot_sweep(
        axes[1],
        rows,
        "heavy_ni_sweep",
        "Heavy Cubic Edge Sweep (heavy_iterations=8)",
        "ni",
        "edge length n (ni=nj=nk=n)",
        experiment_labels("heavy_ni_sweep", analysis_mode, run_set),
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_chunk_sweep_pages(pdf, rows, analysis_mode, run_set, chunk_sweep_ni):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plot_sweep(
        axes[0],
        rows,
        "stencil_chunk_sweep",
        f"Stencil Chunk Sweep (ni=nj=nk={chunk_sweep_ni})",
        "inner_chunk_length",
        "inner chunk length ninner",
        experiment_labels("stencil_chunk_sweep", analysis_mode, run_set),
    )
    plot_sweep(
        axes[1],
        rows,
        "heavy_chunk_sweep",
        f"Heavy Chunk Sweep (ni=nj=nk={chunk_sweep_ni}, heavy_iterations=8)",
        "inner_chunk_length",
        "inner chunk length ninner",
        experiment_labels("heavy_chunk_sweep", analysis_mode, run_set),
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_heavy_intensity_page(pdf, rows, analysis_mode, run_set, chunk_sweep_ni):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plot_sweep(
        axes[0],
        rows,
        "heavy_intensity_sweep",
        f"Heavy Intensity Sweep (ni=nj=nk={chunk_sweep_ni})",
        "heavy_iterations",
        "heavy iterations",
        experiment_labels("heavy_intensity_sweep", analysis_mode, run_set),
    )

    exp_rows = [row for row in rows if row["experiment"] == "heavy_intensity_sweep"]
    by_variant = defaultdict(list)
    for row in exp_rows:
        by_variant[row["variant"]].append(
            (row["arithmetic_intensity_flops_per_byte"], row["updates_per_second"])
        )

    labels = experiment_labels("heavy_intensity_sweep", analysis_mode, run_set)
    for variant, samples in sorted(by_variant.items()):
        samples.sort()
        xs = [x for x, _ in samples]
        ys = [y for _, y in samples]
        color, marker = VARIANT_STYLE.get(variant, ("#333333", "o"))
        axes[1].plot(
            xs, ys, marker=marker, color=color, label=labels.get(variant, variant)
        )

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("estimated arithmetic intensity [flop/byte]")
    axes[1].set_ylabel("updates/s")
    axes[1].set_title("Heavy Roofline Proxy")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def ratio_pairs(run_set):
    if run_set == "gpu":
        return [
            ("kokkos_mdrange_kji", "kokkos_flat_kji"),
            ("gpu_rawspan_chunk_ninj", "kokkos_flat_kji"),
            ("gpu_rawspan_chunk_ni", "kokkos_flat_kji"),
            ("gpu_rawspan_view_chunk_ninj", "kokkos_flat_kji"),
            ("gpu_rawspan_view_chunk_ni", "kokkos_flat_kji"),
            ("gpu_logical_chunk_ninj", "kokkos_flat_kji"),
            ("gpu_logical_chunk_ni", "kokkos_flat_kji"),
        ]
    return [
        ("cpu_rawspan_voi", "cpu_logical_kji"),
        ("cpu_rawspan_ovi", "cpu_logical_kji"),
        ("cpu_logical_ovi", "cpu_logical_kji"),
        ("kokkos_rawspan_ovi", "cpu_logical_kji"),
        ("kokkos_logical_ovi", "cpu_logical_kji"),
    ]


def chunk_ratio_pairs(run_set):
    if run_set == "gpu":
        return [
            ("gpu_rawspan_sweep", "kokkos_flat_kji"),
            ("gpu_rawspan_view_sweep", "kokkos_flat_kji"),
            ("gpu_logical_sweep", "kokkos_flat_kji"),
        ]
    return [
        ("cpu_rawspan_voi", "cpu_logical_kji"),
        ("cpu_rawspan_ovi", "cpu_logical_kji"),
        ("cpu_logical_ovi", "cpu_logical_kji"),
        ("kokkos_rawspan_ovi", "cpu_logical_kji"),
        ("kokkos_logical_ovi", "cpu_logical_kji"),
    ]


def add_ratio_page(pdf, rows, analysis_mode, run_set):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    experiments = [
        ("stencil_ni_sweep", "Stencil Cubic Edge Ratios"),
        ("heavy_ni_sweep", "Heavy Cubic Edge Ratios"),
    ]
    for ax, (experiment, title) in zip(axes, experiments):
        labels = experiment_labels(experiment, analysis_mode, run_set)
        exp_rows = [row for row in rows if row["experiment"] == experiment]
        by_key = {
            (row["variant"], row["ni"]): row["updates_per_second"] for row in exp_rows
        }
        nis = sorted({row["ni"] for row in exp_rows})
        for variant, baseline_variant in ratio_pairs(run_set):
            ratios = []
            valid_nis = []
            for ni in nis:
                if (variant, ni) not in by_key or (baseline_variant, ni) not in by_key:
                    continue
                valid_nis.append(ni)
                ratios.append(by_key[(variant, ni)] / by_key[(baseline_variant, ni)])
            if not valid_nis:
                continue
            ax.plot(
                valid_nis,
                ratios,
                marker=VARIANT_STYLE[variant][1],
                color=VARIANT_STYLE[variant][0],
                label=f"{labels[variant]} / {labels[baseline_variant]}",
            )
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("edge length n (ni=nj=nk=n)")
        ax.set_ylabel("throughput ratio")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_chunk_ratio_page(pdf, rows, analysis_mode, run_set):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    experiments = [
        ("stencil_chunk_sweep", "Stencil Chunk Ratios"),
        ("heavy_chunk_sweep", "Heavy Chunk Ratios"),
    ]
    for ax, (experiment, title) in zip(axes, experiments):
        labels = experiment_labels(experiment, analysis_mode, run_set)
        exp_rows = [row for row in rows if row["experiment"] == experiment]
        by_variant = defaultdict(list)
        for row in exp_rows:
            by_variant[row["variant"]].append(
                (row["inner_chunk_length"], row["updates_per_second"])
            )

        baseline_values = {
            row["variant"]: row["updates_per_second"]
            for row in exp_rows
            if row["variant"] in {pair[1] for pair in chunk_ratio_pairs(run_set)}
        }
        for variant, baseline_variant in chunk_ratio_pairs(run_set):
            samples = sorted(by_variant.get(variant, []))
            baseline = baseline_values.get(baseline_variant, 0.0)
            if not samples or baseline == 0.0:
                continue
            chunks = [chunk for chunk, _ in samples]
            ratios = [ups / baseline for _, ups in samples]
            color, marker = VARIANT_STYLE[variant]
            ax.plot(
                chunks,
                ratios,
                marker=marker,
                color=color,
                label=f"{labels[variant]} / {labels[baseline_variant]}",
            )

        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("inner chunk length ninner")
        ax.set_ylabel("throughput ratio")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_variant_map_page(pdf, args, labels):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    base_variants, chunk_variants, baseline_variants = analysis_variants(
        "gpu" if args.gpu else "cpu"
    )
    lines = ["Variant Map", "", "PDF label -> benchmark flag", ""]
    for variant in base_variants:
        suffix = ""
        if variant in chunk_variants or variant == "kokkos_rawspan_ovi":
            suffix = f" {chunk_flag_text(variant, args.analysis_mode)}"
        lines.append(f"{labels[variant]} -> --variant {variant}{suffix}")
    lines.extend(
        [
            "",
            "Notes",
            "- Dense baseline variants traverse the full allocated memory box (including ghosts).",
            (
                "- In verify mode the script sets --inner-chunk-length <ni> for the CPU and"
                " Kokkos chunked variants during the edge sweep."
            ),
            (
                "- In default mode the raw-span curves use the benchmark's"
                " normal default chunking."
            ),
            f"- Here, {default_chunk_description(args)}.",
            (
                f"- The chunk sweep fixes ni=nj=nk={args.chunk_sweep_ni} and runs explicit chunk lengths"
                " for the chunked variants, with baseline variants included once."
            ),
            f"- Chunked variants in this run: {', '.join(chunk_variants)}.",
            f"- Baseline variants in this run: {', '.join(baseline_variants)}.",
            (
                "- For GPU runs, the report uses a single Kokkos raw-span implementation and"
                " distinguishes cases by explicit inner chunk length."
            ),
        ]
    )
    ax.text(
        0.05,
        0.97,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = str(Path(args.binary))
    if args.target_total_cells is None:
        args.target_total_cells = default_target_total_cells(binary)
    edge_values = parse_csv_ints(args.edge_values)
    chunk_values = parse_csv_ints(args.chunk_values)
    heavy_iteration_values = parse_csv_ints(args.heavy_iteration_values)
    run_set = "gpu" if args.gpu else "cpu"
    base_variants, chunk_sweep_variants, chunk_sweep_baselines = analysis_variants(
        run_set
    )
    labels = variant_labels(args.analysis_mode, run_set)
    progress = ProgressReporter(
        count_total_runs(
            base_variants,
            chunk_sweep_variants,
            chunk_sweep_baselines,
            edge_values,
            chunk_values,
            heavy_iteration_values,
            not args.skip_heavy_intensity_sweep,
        )
    )

    combined_csv = output_dir / "analysis.csv"
    output_pdf = output_dir / "analysis.pdf"

    rows = []
    print(f"Starting analysis suite: {progress.total_runs} benchmark runs", flush=True)
    rows.extend(
        run_sweep(
            binary,
            combined_csv,
            "stencil",
            base_variants,
            edge_values,
            args,
            args.analysis_mode,
            progress,
        )
    )
    rows.extend(
        run_sweep(
            binary,
            combined_csv,
            "heavy",
            base_variants,
            edge_values,
            args,
            args.analysis_mode,
            progress,
        )
    )
    rows.extend(
        run_chunk_sweep(
            binary,
            combined_csv,
            "stencil",
            chunk_sweep_variants,
            chunk_sweep_baselines,
            chunk_values,
            args,
            progress,
        )
    )
    rows.extend(
        run_chunk_sweep(
            binary,
            combined_csv,
            "heavy",
            chunk_sweep_variants,
            chunk_sweep_baselines,
            chunk_values,
            args,
            progress,
        )
    )
    if not args.skip_heavy_intensity_sweep:
        rows.extend(
            run_heavy_intensity_sweep(
                binary,
                combined_csv,
                base_variants,
                args.chunk_sweep_ni,
                heavy_iteration_values,
                args,
                progress,
            )
        )
    write_combined_csv(combined_csv, rows)

    system_info = collect_system_info(binary)
    with PdfPages(output_pdf) as pdf:
        add_title_page(
            pdf,
            args.title,
            args,
            system_info,
            rows,
            labels,
            chunk_values,
            heavy_iteration_values,
        )
        add_variant_map_page(pdf, args, labels)
        add_sweep_pages(pdf, rows, args.analysis_mode, run_set)
        add_ratio_page(pdf, rows, args.analysis_mode, run_set)
        add_chunk_sweep_pages(
            pdf, rows, args.analysis_mode, run_set, args.chunk_sweep_ni
        )
        add_chunk_ratio_page(pdf, rows, args.analysis_mode, run_set)
        if not args.skip_heavy_intensity_sweep:
            add_heavy_intensity_page(
                pdf, rows, args.analysis_mode, run_set, args.chunk_sweep_ni
            )

    print(f"Wrote {combined_csv}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
