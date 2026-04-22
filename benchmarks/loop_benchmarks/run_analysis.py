#!/usr/bin/env python3

import argparse
import csv
import os
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "parthenon-loop-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


BASE_VARIANTS = [
    "cpu_simd",
    "cpu_coalesced_outer_var",
    "cpu_rowvar_simd",
    "cpu_hierarchical",
    "hierarchical",
]

CHUNK_SWEEP_VARIANTS = [
    "cpu_coalesced_outer_var",
    "cpu_hierarchical",
    "hierarchical",
]

VARIANT_STYLE = {
    "cpu_simd": ("#0f4c5c", "o"),
    "cpu_coalesced_outer_var": ("#2a9d8f", "P"),
    "cpu_rowvar_simd": ("#355070", "D"),
    "cpu_hierarchical": ("#b56576", "s"),
    "hierarchical": ("#6d597a", "^"),
}


def variant_labels(analysis_mode):
    raw_chunk_label = "chunk=ni" if analysis_mode == "verify" else "default chunks"
    return {
        "cpu_simd": "CPU SIMD, (v, outer, inner)",
        "cpu_coalesced_outer_var": f"CPU raw-span, {raw_chunk_label}, (v, outer, inner)",
        "cpu_rowvar_simd": "CPU SIMD, (outer, v, inner)",
        "cpu_hierarchical": f"CPU raw-span, {raw_chunk_label}, (outer, v, inner)",
        "hierarchical": "Kokkos raw-span, default chunks",
    }


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
    return parser.parse_args()


def run_command(cmd):
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return completed.stdout


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


def collect_system_info():
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    commands = [
        ("uname", ["uname", "-a"]),
        ("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
    ]
    for key, cmd in commands:
        try:
            info[key] = run_command(cmd).strip()
        except Exception:
            continue
    return info


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


def run_sweep(binary, csv_path, kernel, variants, ni_values, args, analysis_mode):
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
            rows.append(numericize(summary))
    return rows


def run_chunk_sweep(binary, csv_path, kernel, variants, chunk_values, args):
    rows = []
    ni = args.chunk_sweep_ni
    for chunk_length in chunk_values:
        for variant in variants:
            cmd = build_benchmark_command(binary, csv_path, kernel, variant, ni, args, chunk_length)
            summary = parse_summary(run_command(cmd))
            summary["repeats"] = str(args.repeats)
            summary["experiment"] = f"{kernel}_chunk_sweep"
            rows.append(numericize(summary))
    for variant in ("cpu_simd", "cpu_rowvar_simd"):
        cmd = build_benchmark_command(binary, csv_path, kernel, variant, ni, args)
        summary = parse_summary(run_command(cmd))
        summary["repeats"] = str(args.repeats)
        summary["experiment"] = f"{kernel}_chunk_sweep"
        rows.append(numericize(summary))
    return rows


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
        "",
        "Experiments:",
        (
            f"- stencil ni sweep: {labels['cpu_simd']} vs "
            f"{labels['cpu_coalesced_outer_var']} vs "
            f"{labels['cpu_rowvar_simd']} vs {labels['cpu_hierarchical']} vs "
            f"{labels['hierarchical']}"
        ),
        (
            f"- heavy ni sweep: {labels['cpu_simd']} vs "
            f"{labels['cpu_coalesced_outer_var']} vs "
            f"{labels['cpu_rowvar_simd']} vs {labels['cpu_hierarchical']} vs "
            f"{labels['hierarchical']}"
        ),
        (
            f"- chunk sweeps at ni={args.chunk_sweep_ni}: raw-span variants and "
            "Kokkos across explicit inner chunk lengths, with SIMD baselines included"
        ),
    ]
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


def add_ratio_page(pdf, rows, labels):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    experiments = [
        ("stencil_ni_sweep", "Stencil Ratios"),
        ("heavy_ni_sweep", "Heavy Ratios"),
    ]
    for ax, (experiment, title) in zip(axes, experiments):
        exp_rows = [row for row in rows if row["experiment"] == experiment]
        by_key = {(row["variant"], row["ni"]): row["updates_per_second"] for row in exp_rows}
        nis = sorted({row["ni"] for row in exp_rows})
        coalesced_ratio = []
        rowvar_ratio = []
        cpu_hier_ratio = []
        hier_ratio = []
        for ni in nis:
            baseline = by_key[("cpu_simd", ni)]
            coalesced = by_key[("cpu_coalesced_outer_var", ni)]
            rowvar = by_key[("cpu_rowvar_simd", ni)]
            cpu_hier = by_key[("cpu_hierarchical", ni)]
            kokkos_hier = by_key[("hierarchical", ni)]
            coalesced_ratio.append(coalesced / baseline)
            rowvar_ratio.append(rowvar / baseline)
            cpu_hier_ratio.append(cpu_hier / baseline)
            hier_ratio.append(kokkos_hier / baseline)
        ax.plot(
            nis,
            coalesced_ratio,
            marker=VARIANT_STYLE["cpu_coalesced_outer_var"][1],
            color=VARIANT_STYLE["cpu_coalesced_outer_var"][0],
            label=f"{labels['cpu_coalesced_outer_var']} / {labels['cpu_simd']}",
        )
        ax.plot(
            nis,
            rowvar_ratio,
            marker=VARIANT_STYLE["cpu_rowvar_simd"][1],
            color=VARIANT_STYLE["cpu_rowvar_simd"][0],
            label=f"{labels['cpu_rowvar_simd']} / {labels['cpu_simd']}",
        )
        ax.plot(
            nis,
            cpu_hier_ratio,
            marker=VARIANT_STYLE["cpu_hierarchical"][1],
            color=VARIANT_STYLE["cpu_hierarchical"][0],
            label=f"{labels['cpu_hierarchical']} / {labels['cpu_simd']}",
        )
        ax.plot(
            nis,
            hier_ratio,
            marker=VARIANT_STYLE["hierarchical"][1],
            color=VARIANT_STYLE["hierarchical"][0],
            label=f"{labels['hierarchical']} / {labels['cpu_simd']}",
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


def add_chunk_ratio_page(pdf, rows, labels):
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

        simd_v = exp_rows[0]["updates_per_second"] if exp_rows else 0.0
        simd_outer = simd_v
        for row in exp_rows:
            if row["variant"] == "cpu_simd":
                simd_v = row["updates_per_second"]
            elif row["variant"] == "cpu_rowvar_simd":
                simd_outer = row["updates_per_second"]

        plotted = [
            ("cpu_coalesced_outer_var", simd_v, "cpu_simd"),
            ("cpu_hierarchical", simd_outer, "cpu_rowvar_simd"),
            ("hierarchical", simd_v, "cpu_simd"),
        ]
        for variant, baseline, baseline_variant in plotted:
            samples = sorted(by_variant.get(variant, []))
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
    lines = [
        "Variant Map",
        "",
        "PDF label -> benchmark flag",
        "",
        f"{labels['cpu_simd']} -> --variant cpu_simd",
        f"{labels['cpu_coalesced_outer_var']} -> --variant cpu_coalesced_outer_var {chunk_flag_text('cpu_coalesced_outer_var', args.analysis_mode)}",
        f"{labels['cpu_rowvar_simd']} -> --variant cpu_rowvar_simd",
        f"{labels['cpu_hierarchical']} -> --variant cpu_hierarchical {chunk_flag_text('cpu_hierarchical', args.analysis_mode)}",
        f"{labels['hierarchical']} -> --variant hierarchical {chunk_flag_text('hierarchical', args.analysis_mode)}",
        "",
        "Notes",
        (
            "- In verify mode the script sets --inner-chunk-length <ni> for both CPU raw-span"
            " variants during the ni sweep."
        ),
        (
            "- In default mode the CPU and Kokkos raw-span curves use the benchmark's"
            " normal default chunking."
        ),
        f"- Here, {default_chunk_description(args)}.",
        (
            f"- The chunk sweep fixes ni={args.chunk_sweep_ni} and runs explicit chunk lengths"
            " for the raw-span variants."
        ),
    ]
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
    labels = variant_labels(args.analysis_mode)

    combined_csv = output_dir / "analysis.csv"
    output_pdf = output_dir / "analysis.pdf"

    rows = []
    rows.extend(run_sweep(binary, combined_csv, "stencil", BASE_VARIANTS, ni_values, args, args.analysis_mode))
    rows.extend(run_sweep(binary, combined_csv, "heavy", BASE_VARIANTS, ni_values, args, args.analysis_mode))
    rows.extend(run_chunk_sweep(binary, combined_csv, "stencil", CHUNK_SWEEP_VARIANTS, chunk_values, args))
    rows.extend(run_chunk_sweep(binary, combined_csv, "heavy", CHUNK_SWEEP_VARIANTS, chunk_values, args))
    write_combined_csv(combined_csv, rows)

    system_info = collect_system_info()
    with PdfPages(output_pdf) as pdf:
        add_title_page(pdf, args.title, args, system_info, rows, labels, chunk_values)
        add_variant_map_page(pdf, args, labels)
        add_sweep_pages(pdf, rows, labels)
        add_ratio_page(pdf, rows, labels)
        add_chunk_sweep_pages(pdf, rows, labels, args.chunk_sweep_ni)
        add_chunk_ratio_page(pdf, rows, labels)

    print(f"Wrote {combined_csv}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
