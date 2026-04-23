#!/usr/bin/env python3

import argparse
import csv
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "parthenon-loop-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


VARIANT_STYLE = {
    "flat": ("#264653", "o"),
    "mdrange": ("#287271", "D"),
    "tuned": ("#8ab17d", "P"),
    "cpu_simd": ("#0f4c5c", "o"),
    "cpu_coalesced_outer_var": ("#2a9d8f", "P"),
    "cpu_rowvar_simd": ("#355070", "D"),
    "cpu_hierarchical": ("#b56576", "s"),
    "hierarchical": ("#6d597a", "^"),
}

CPU_BASE_VARIANTS = [
    "cpu_simd",
    "cpu_coalesced_outer_var",
    "cpu_rowvar_simd",
    "cpu_hierarchical",
    "hierarchical",
]

CPU_CHUNK_SWEEP_VARIANTS = [
    "cpu_coalesced_outer_var",
    "cpu_hierarchical",
    "hierarchical",
]

CPU_CHUNK_SWEEP_BASELINES = ["cpu_simd", "cpu_rowvar_simd"]

GPU_BASE_VARIANTS = [
    "flat",
    "mdrange",
    "hierarchical",
    "tuned",
]

GPU_CHUNK_SWEEP_VARIANTS = [
    "hierarchical",
    "tuned",
]

GPU_CHUNK_SWEEP_BASELINES = ["flat", "mdrange"]


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
            f"[{self.completed_runs}/{self.total_runs}] {experiment}: " + ", ".join(details),
            flush=True,
        )


def variant_labels(analysis_mode, run_set):
    raw_chunk_label = "chunk=ni" if analysis_mode == "verify" else "default chunks"
    cpu_labels = {
        "cpu_simd": "CPU SIMD, (v, outer, inner)",
        "cpu_coalesced_outer_var": f"CPU raw-span, {raw_chunk_label}, (v, outer, inner)",
        "cpu_rowvar_simd": "CPU SIMD, (outer, v, inner)",
        "cpu_hierarchical": f"CPU raw-span, {raw_chunk_label}, (outer, v, inner)",
        "hierarchical": "Kokkos raw-span, default chunks",
    }
    if run_set == "gpu":
        return {
            "flat": "Kokkos flat range",
            "mdrange": "Kokkos MDRange",
            "hierarchical": "Kokkos raw-span, default chunks",
            "tuned": "Kokkos tuned raw-span",
        }
    return cpu_labels


def cpu_raw_span_chunk_mode(analysis_mode):
    return "row" if analysis_mode == "verify" else None


def chunk_flag_text(variant, analysis_mode):
    if variant in {"cpu_coalesced_outer_var", "cpu_hierarchical"}:
        return "--inner-chunk-length <ni>" if analysis_mode == "verify" else "(default chunks)"
    if variant == "hierarchical":
        return "(default chunks)"
    return ""


def default_chunk_description(args):
    return (
        "default inner chunk length = ni*nj "
        f"(so at fixed chunk-sweep ni={args.chunk_sweep_ni}, default={args.chunk_sweep_ni * args.nj})"
    )


def analysis_variants(run_set):
    if run_set == "gpu":
        return GPU_BASE_VARIANTS, GPU_CHUNK_SWEEP_VARIANTS, GPU_CHUNK_SWEEP_BASELINES
    return CPU_BASE_VARIANTS, CPU_CHUNK_SWEEP_VARIANTS, CPU_CHUNK_SWEEP_BASELINES


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
    parser.add_argument("--nk", type=int, default=32, help="k extent")
    parser.add_argument("--nj", type=int, default=32, help="j extent")
    parser.add_argument("--ghosts", type=int, default=2, help="Ghost zone width")
    parser.add_argument(
        "--ni-values",
        default="8,16,32,64,128",
        help="Comma-separated ni values to sweep",
    )
    parser.add_argument(
        "--target-block-ni-product",
        type=int,
        default=512,
        help="Choose blocks so blocks * ni stays fixed at this value",
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
        help="Comma-separated inner chunk lengths to sweep at fixed ni",
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
        "repeats",
        "total_updates",
    }
    float_fields = {
        "min_seconds",
        "median_seconds",
        "mean_seconds",
        "updates_per_second",
        "estimated_bandwidth_gb_s",
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
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
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
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        key, _sep, _value_type = key_type.partition(":")
        cache[key] = value
    return cache


def parse_flags_make(flags_make_path):
    info = {}
    if not flags_make_path.exists():
        return info
    for line in flags_make_path.read_text(encoding="utf-8", errors="ignore").splitlines():
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
    info["cmake_cxx_flags"] = " ".join(part for part in [common_flags, build_flags] if part)

    flags_make = build_dir / "benchmarks/loop_benchmarks/CMakeFiles/loop-benchmarks.dir/flags.make"
    target_flags = parse_flags_make(flags_make)
    info["target_compile_flags"] = target_flags.get("CXX_FLAGS", "")
    info["target_defines"] = target_flags.get("CXX_DEFINES", "")
    return info


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


def build_benchmark_command(binary, csv_path, kernel, variant, ni, args, chunk_length=None):
    cmd = [
        binary,
        "--kernel",
        kernel,
        "--variant",
        variant,
        "--blocks",
        str(max(1, args.target_block_ni_product // ni)),
        "--vars",
        str(args.vars),
        "--nk",
        str(args.nk),
        "--nj",
        str(args.nj),
        "--ni",
        str(ni),
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


def run_sweep(binary, csv_path, kernel, variants, ni_values, args, analysis_mode, progress):
    rows = []
    raw_chunk_mode = cpu_raw_span_chunk_mode(analysis_mode)
    for ni in ni_values:
        for variant in variants:
            chunk_length = None
            if variant in {"cpu_coalesced_outer_var", "cpu_hierarchical"} and raw_chunk_mode == "row":
                chunk_length = ni
            cmd = build_benchmark_command(binary, csv_path, kernel, variant, ni, args, chunk_length)
            summary = parse_summary(run_command(cmd))
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
            cmd = build_benchmark_command(binary, csv_path, kernel, variant, ni, args, chunk_length)
            summary = parse_summary(run_command(cmd))
            summary["repeats"] = str(args.repeats)
            summary["experiment"] = f"{kernel}_chunk_sweep"
            row = numericize(summary)
            rows.append(row)
            progress.log_completed(summary["experiment"], row)
    for variant in baseline_variants:
        cmd = build_benchmark_command(binary, csv_path, kernel, variant, ni, args)
        summary = parse_summary(run_command(cmd))
        summary["repeats"] = str(args.repeats)
        summary["experiment"] = f"{kernel}_chunk_sweep"
        row = numericize(summary)
        rows.append(row)
        progress.log_completed(summary["experiment"], row)
    return rows


def count_total_runs(base_variants, chunk_sweep_variants, chunk_sweep_baselines, ni_values, chunk_values):
    return (
        2 * len(base_variants) * len(ni_values)
        + 2 * len(chunk_sweep_variants) * len(chunk_values)
        + 2 * len(chunk_sweep_baselines)
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
        "repeats",
        "min_seconds",
        "median_seconds",
        "mean_seconds",
        "updates_per_second",
        "estimated_bandwidth_gb_s",
        "total_updates",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_title_page(pdf, title, args, system_info, rows, labels, chunk_values):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ni_values = sorted({row["ni"] for row in rows})
    lines = [
        title,
        "",
        f"Binary: {args.binary}",
        f"variant set: {'gpu' if args.gpu else 'cpu'}",
        f"analysis mode: {args.analysis_mode}",
        f"ni sweep: {ni_values}",
        f"chunk sweep: ni={args.chunk_sweep_ni}, chunks={chunk_values}",
        f"shape base: vars={args.vars}, nk={args.nk}, nj={args.nj}, ghosts={args.ghosts}",
        default_chunk_description(args),
        f"block*ni target: {args.target_block_ni_product}",
        f"repeats={args.repeats}, warmup={args.warmup}, heavy_iterations={args.heavy_iterations}",
        "",
        f"Platform: {system_info.get('platform', 'unknown')}",
        f"CPU: {system_info.get('cpu', 'unknown')}",
        f"uname: {system_info.get('uname', 'unknown')}",
    ]
    append_wrapped_line(lines, "Build dir: ", system_info.get("build_dir", ""))
    append_wrapped_line(lines, "Build type: ", system_info.get("build_type", ""))
    append_wrapped_line(lines, "Compiler: ", system_info.get("compiler", ""))
    append_wrapped_line(lines, "Compiler version: ", system_info.get("compiler_version", ""))
    append_wrapped_line(lines, "Target compile flags: ", system_info.get("target_compile_flags", ""))
    append_wrapped_line(lines, "Target defines: ", system_info.get("target_defines", ""))
    append_wrapped_line(lines, "CMake CXX flags: ", system_info.get("cmake_cxx_flags", ""))
    lines.extend(
        [
            "",
            "Experiments:",
            ("- stencil ni sweep: " + " vs ".join(labels[row["variant"]] for row in rows if row["experiment"] == "stencil_ni_sweep" and row["ni"] == ni_values[0]))
            if ni_values
            else "",
            ("- heavy ni sweep: " + " vs ".join(labels[row["variant"]] for row in rows if row["experiment"] == "heavy_ni_sweep" and row["ni"] == ni_values[0]))
            if ni_values
            else "",
            (
                f"- chunk sweeps at ni={args.chunk_sweep_ni}: raw-span variants and "
                "selected baselines across explicit inner chunk lengths"
            ),
        ]
    )
    ax.text(0.05, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
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


def add_sweep_pages(pdf, rows, labels):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plot_sweep(axes[0], rows, "stencil_ni_sweep", "Stencil ni Sweep", "ni", "ni", labels)
    plot_sweep(
        axes[1],
        rows,
        "heavy_ni_sweep",
        "Heavy ni Sweep (heavy_iterations=8)",
        "ni",
        "ni",
        labels,
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_chunk_sweep_pages(pdf, rows, labels, chunk_sweep_ni):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plot_sweep(
        axes[0],
        rows,
        "stencil_chunk_sweep",
        f"Stencil Chunk Sweep (ni={chunk_sweep_ni})",
        "inner_chunk_length",
        "inner chunk length",
        labels,
    )
    plot_sweep(
        axes[1],
        rows,
        "heavy_chunk_sweep",
        f"Heavy Chunk Sweep (ni={chunk_sweep_ni}, heavy_iterations=8)",
        "inner_chunk_length",
        "inner chunk length",
        labels,
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def ratio_pairs(run_set):
    if run_set == "gpu":
        return [("mdrange", "flat"), ("hierarchical", "flat"), ("tuned", "flat")]
    return [
        ("cpu_coalesced_outer_var", "cpu_simd"),
        ("cpu_rowvar_simd", "cpu_simd"),
        ("cpu_hierarchical", "cpu_simd"),
        ("hierarchical", "cpu_simd"),
    ]


def chunk_ratio_pairs(run_set):
    if run_set == "gpu":
        return [("hierarchical", "flat"), ("tuned", "flat")]
    return [
        ("cpu_coalesced_outer_var", "cpu_simd"),
        ("cpu_hierarchical", "cpu_rowvar_simd"),
        ("hierarchical", "cpu_simd"),
    ]


def add_ratio_page(pdf, rows, labels, run_set):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    experiments = [
        ("stencil_ni_sweep", "Stencil Ratios"),
        ("heavy_ni_sweep", "Heavy Ratios"),
    ]
    for ax, (experiment, title) in zip(axes, experiments):
        exp_rows = [row for row in rows if row["experiment"] == experiment]
        by_key = {(row["variant"], row["ni"]): row["updates_per_second"] for row in exp_rows}
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
        ax.set_xlabel("ni")
        ax.set_ylabel("throughput ratio")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_chunk_ratio_page(pdf, rows, labels, run_set):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    experiments = [
        ("stencil_chunk_sweep", "Stencil Chunk Ratios"),
        ("heavy_chunk_sweep", "Heavy Chunk Ratios"),
    ]
    for ax, (experiment, title) in zip(axes, experiments):
        exp_rows = [row for row in rows if row["experiment"] == experiment]
        by_variant = defaultdict(list)
        for row in exp_rows:
            by_variant[row["variant"]].append((row["inner_chunk_length"], row["updates_per_second"]))

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
        ax.set_xlabel("inner chunk length")
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
    base_variants, chunk_variants, baseline_variants = analysis_variants("gpu" if args.gpu else "cpu")
    lines = ["Variant Map", "", "PDF label -> benchmark flag", ""]
    for variant in base_variants:
        suffix = ""
        if variant in chunk_variants or variant == "hierarchical":
            suffix = f" {chunk_flag_text(variant, args.analysis_mode)}"
        lines.append(f"{labels[variant]} -> --variant {variant}{suffix}")
    lines.extend(
        [
            "",
            "Notes",
            (
                "- In verify mode the script sets --inner-chunk-length <ni> for the CPU raw-span"
                " variants during the ni sweep."
            ),
            (
                "- In default mode the raw-span curves use the benchmark's"
                " normal default chunking."
            ),
            f"- Here, {default_chunk_description(args)}.",
            (
                f"- The chunk sweep fixes ni={args.chunk_sweep_ni} and runs explicit chunk lengths"
                " for the chunked variants, with baseline variants included once."
            ),
            f"- Chunked variants in this run: {', '.join(chunk_variants)}.",
            f"- Baseline variants in this run: {', '.join(baseline_variants)}.",
        ]
    )
    ax.text(0.05, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = str(Path(args.binary))
    ni_values = [int(part.strip()) for part in args.ni_values.split(",") if part.strip()]
    chunk_values = [int(part.strip()) for part in args.chunk_values.split(",") if part.strip()]
    run_set = "gpu" if args.gpu else "cpu"
    base_variants, chunk_sweep_variants, chunk_sweep_baselines = analysis_variants(run_set)
    labels = variant_labels(args.analysis_mode, run_set)
    progress = ProgressReporter(
        count_total_runs(
            base_variants, chunk_sweep_variants, chunk_sweep_baselines, ni_values, chunk_values
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
            ni_values,
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
            ni_values,
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
    write_combined_csv(combined_csv, rows)

    system_info = collect_system_info(binary)
    with PdfPages(output_pdf) as pdf:
        add_title_page(pdf, args.title, args, system_info, rows, labels, chunk_values)
        add_variant_map_page(pdf, args, labels)
        add_sweep_pages(pdf, rows, labels)
        add_ratio_page(pdf, rows, labels, run_set)
        add_chunk_sweep_pages(pdf, rows, labels, args.chunk_sweep_ni)
        add_chunk_ratio_page(pdf, rows, labels, run_set)

    print(f"Wrote {combined_csv}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
