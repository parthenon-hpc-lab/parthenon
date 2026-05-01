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

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "parthenon-loop-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_LOOPS = [
    "cpu_flat_ghosts",
    "cpu_boiv_contiguous",
    "cpu_bovi_contiguous",
    "cpu_bvoi_contiguous",
    "cpu_boiv_logical",
    "cpu_bovi_logical",
    "cpu_bvoi_logical",
    "kokkos_boiv_flat",
    "kokkos_bovi_team_contiguous",
    "kokkos_bovi_team_logical",
    "loop_abstraction_bovi_memory",
    "loop_abstraction_bovi_logical",
    "loop_abstraction_boiv_logical",
    "loop_abstraction_bvoi_memory",
    "loop_abstraction_bvoi_logical",
]

LOOP_ACCESS_MODES = {
    "cpu_flat_ghosts": ["direct", "hoisted"],
    "cpu_boiv_contiguous": ["direct"],
    "cpu_bovi_contiguous": ["direct", "hoisted"],
    "cpu_bvoi_contiguous": ["direct", "hoisted"],
    "cpu_boiv_logical": ["direct"],
    "cpu_bovi_logical": ["direct"],
    "cpu_bvoi_logical": ["direct"],
    "kokkos_boiv_flat": ["direct"],
    "kokkos_bovi_team_contiguous": ["direct", "hoisted"],
    "kokkos_bovi_team_logical": ["direct"],
    "loop_abstraction_bovi_memory": ["hoisted"],
    "loop_abstraction_bovi_logical": ["hoisted"],
    "loop_abstraction_boiv_logical": ["hoisted"],
    "loop_abstraction_bvoi_memory": ["hoisted"],
    "loop_abstraction_bvoi_logical": ["hoisted"],
}

LOOP_STYLE = {
    # Okabe-Ito colorblind-safe palette, with marker shape carrying extra identity.
    "cpu_flat_ghosts": ("#000000", "o"),
    "cpu_boiv_contiguous": ("#0072B2", "s"),
    "cpu_bovi_contiguous": ("#E69F00", "^"),
    "cpu_bvoi_contiguous": ("#009E73", "D"),
    "cpu_boiv_logical": ("#D55E00", "<"),
    "cpu_bovi_logical": ("#CC79A7", ">"),
    "cpu_bvoi_logical": ("#56B4E9", "P"),
    "kokkos_boiv_flat": ("#F0E442", "X"),
    "kokkos_bovi_team_contiguous": ("#882255", "v"),
    "kokkos_bovi_team_logical": ("#44AA99", "*"),
    "loop_abstraction_bovi_memory": ("#999999", "h"),
    "loop_abstraction_bovi_logical": ("#6A3D9A", "p"),
    "loop_abstraction_boiv_logical": ("#B15928", "8"),
    "loop_abstraction_bvoi_memory": ("#1B9E77", "H"),
    "loop_abstraction_bvoi_logical": ("#D95F02", "d"),
}

ACCESS_STYLE = {
    "direct": "-",
    "hoisted": "--",
}


def parse_csv_ints(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_stencil_shapes(text):
    aliases = {
        "point": ("0", "0", "0"),
        "x3": ("-1;0;1", "0", "0"),
        "y3": ("0", "-1;0;1", "0"),
        "z3": ("0", "0", "-1;0;1"),
    }
    shapes = []
    for part in text.split(","):
        name = part.strip()
        if not name:
            continue
        if name in aliases:
            shapes.append(aliases[name])
            continue
        pieces = name.split("/")
        if len(pieces) != 3:
            raise ValueError(f"bad stencil shape '{name}'")
        shapes.append(tuple(pieces))
    return shapes


def parse_args():
    parser = argparse.ArgumentParser(description="Run the v2 loop benchmark analysis.")
    parser.add_argument("--analysis-mode", choices=["standard", "ninner", "full"], default="standard")
    parser.add_argument("--binary", default="build-make/benchmarks/loop_benchmarks_v2/loop-benchmarks-v2")
    parser.add_argument("--output-dir", default="reports/loop_benchmarks_v2")
    parser.add_argument("--title", default="Parthenon Loop Benchmark v2")
    parser.add_argument("--loops", default=",".join(DEFAULT_LOOPS))
    parser.add_argument("--edge-values", default="8,32,128")
    parser.add_argument(
        "--ninner-values",
        default="",
        help="Comma-separated ninner values. Defaults to edge^2 for each block edge.",
    )
    parser.add_argument("--niter-values", default="1,16,128")
    parser.add_argument("--target-total-cells", type=int, default=1_048_576)
    parser.add_argument("--nvars", type=int, default=16)
    parser.add_argument("--nghost", type=int, default=2)
    parser.add_argument("--stencil-x", default="0")
    parser.add_argument("--stencil-y", default="0")
    parser.add_argument("--stencil-z", default="0")
    parser.add_argument(
        "--stencil-shapes",
        default="",
        help="Comma-separated aliases point,x3,y3,z3 or explicit x/y/z offset strings.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", default="Serial")
    return parser.parse_args()


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


def try_run_command(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def detect_cpu_description():
    candidates = []
    system = platform.system()
    if system == "Darwin":
        hardware = try_run_command(["system_profiler", "SPHardwareDataType"])
        chip = ""
        model = ""
        for line in hardware.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "Chip":
                chip = value
            elif key == "Model Name":
                model = value
        if chip and model:
            candidates.append(f"{model} ({chip})")
        elif chip:
            candidates.append(chip)
        elif model:
            candidates.append(model)
        candidates.append(try_run_command(["sysctl", "-n", "machdep.cpu.brand_string"]))
    elif system == "Linux":
        lscpu = try_run_command(["lscpu"])
        for line in lscpu.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip() == "Model name" and value.strip():
                candidates.append(value.strip())
                break

    candidates.extend([platform.processor().strip(), platform.machine().strip()])
    for candidate in candidates:
        if candidate:
            return candidate
    return "unknown"


def collect_metadata(binary):
    meta = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": detect_cpu_description(),
        "uname": try_run_command(["uname", "-a"]) or "unknown",
    }
    build_dir = find_build_dir(binary)
    if build_dir is None:
        return meta

    cache = parse_cmake_cache(build_dir / "CMakeCache.txt")
    flags = parse_flags_make(
        build_dir / "benchmarks/loop_benchmarks_v2/CMakeFiles/loop-benchmarks-v2.dir/flags.make"
    )
    compiler = cache.get("CMAKE_CXX_COMPILER", "unknown")
    meta["compiler"] = compiler
    meta["compiler_id"] = cache.get("CMAKE_CXX_COMPILER_ID", "unknown")
    meta["compiler_version"] = cache.get("CMAKE_CXX_COMPILER_VERSION", "unknown")
    meta["kokkos_backends"] = ", ".join(
        key.replace("Kokkos_ENABLE_", "")
        for key in ("Kokkos_ENABLE_SERIAL", "Kokkos_ENABLE_CUDA", "Kokkos_ENABLE_HIP")
        if cache.get(key, "").upper() == "ON"
    ) or "unknown"
    meta["cxx_flags"] = " ".join(value for key, value in flags.items() if key.startswith("CXX_")).strip()
    compiler_version = try_run_command([compiler, "--version"]) if compiler != "unknown" else ""
    if compiler_version:
        meta["compiler_version_text"] = compiler_version.splitlines()[0]
    return meta


def write_cases_csv(path, loops, edge_values, ninner_values, stencil_shapes, args):
    niter_values = parse_csv_ints(args.niter_values)

    def kernel_label(niter, stencil_x, stencil_y, stencil_z):
        offsets = f"offsets=x{{{stencil_x}}}y{{{stencil_y}}}z{{{stencil_z}}}"
        return f"niter={niter},{offsets}"

    def is_pointwise(stencil):
        return stencil == ("0", "0", "0")

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "loop",
                "access_mode",
                "kernel_label",
                "backend",
                "nblocks",
                "target_cells",
                "nvars",
                "nz",
                "ny",
                "nx",
                "nghost",
                "ninner",
                "niter",
                "stencil_x",
                "stencil_y",
                "stencil_z",
                "warmup",
                "repeats",
                "vars_per_block",
            ],
        )
        writer.writeheader()
        for edge in edge_values:
            edge_ninner_values = ninner_values if ninner_values else [edge * edge]
            for loop in loops:
                loop_stencil_shapes = [
                    stencil for stencil in stencil_shapes
                    if loop != "cpu_flat_ghosts" or is_pointwise(stencil)
                ]
                for access_mode in LOOP_ACCESS_MODES.get(loop, ["direct"]):
                    for ninner in edge_ninner_values:
                        for stencil_x, stencil_y, stencil_z in loop_stencil_shapes:
                            for niter in niter_values:
                                writer.writerow(
                                    {
                                        "loop": loop,
                                        "access_mode": access_mode,
                                        "kernel_label": kernel_label(
                                            niter, stencil_x, stencil_y, stencil_z
                                        ),
                                        "backend": args.backend,
                                        "nblocks": 0,
                                        "target_cells": args.target_total_cells,
                                        "nvars": args.nvars,
                                        "nz": edge,
                                        "ny": edge,
                                        "nx": edge,
                                        "nghost": args.nghost,
                                        "ninner": ninner,
                                        "niter": niter,
                                        "stencil_x": stencil_x,
                                        "stencil_y": stencil_y,
                                        "stencil_z": stencil_z,
                                        "warmup": args.warmup,
                                        "repeats": args.repeats,
                                        "vars_per_block": "",
                                    }
                                )


def run_binary(binary, cases_csv, results_csv):
    cases_csv = Path(cases_csv)
    results_csv = Path(results_csv)
    if results_csv.exists():
        results_csv.unlink()
    subprocess.run([binary, "--cases", cases_csv, "--csv-out", results_csv], check=True)


def read_results_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_number(value):
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except Exception:
        return value


def numericize_rows(rows):
    return [{k: to_number(v) for k, v in row.items()} for row in rows]


def row_edge(row):
    return row.get("nx_interior", row.get("nx"))


def row_blocks(row):
    return row.get("nblocks", row.get("blocks"))


def row_kernel(row):
    return row.get("kernel_label", "unknown")


def row_access_mode(row):
    return row.get("access_mode", "direct")


def series_key(row, x_key="edge"):
    if x_key == "ninner":
        return (row_kernel(row), row["loop"], row_access_mode(row), row_edge(row))
    return (row_kernel(row), row["loop"], row_access_mode(row))


def series_label(row, include_kernel=False, include_edge=False):
    label = f"{row['loop']} [{row_access_mode(row)}]"
    if include_edge:
        label = f"{label}, edge={row_edge(row)}"
    if include_kernel:
        return f"{row_kernel(row)} | {label}"
    return label


def wrap_lines(items, width=94):
    lines = []
    for item in items:
        if item == "":
            lines.append("")
        else:
            lines.extend(
                textwrap.wrap(item, width=width, break_long_words=False, break_on_hyphens=False)
            )
    return lines


def add_text_page(pdf, title, lines):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, pad=12)
    ax.text(
        0.05,
        0.97,
        "\n".join(wrap_lines(lines)),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10.5,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def row_x(row, x_key):
    if x_key == "ninner":
        return row["ninner"]
    return row_edge(row)


def plot_series(ax, rows, y_key, y_label, title, x_key="edge"):
    by_loop = defaultdict(list)
    for row in rows:
        by_loop[series_key(row, x_key)].append(row)

    for key, points in sorted(by_loop.items()):
        points = sorted(points, key=lambda row: row_x(row, x_key))
        xs = [row_x(row, x_key) for row in points]
        ys = [row[y_key] for row in points]
        _kernel, loop, access_mode = key[:3]
        color, marker = LOOP_STYLE.get(loop, ("#333333", "o"))
        linestyle = ACCESS_STYLE.get(access_mode, "-")
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=5,
            label=series_label(points[0], include_kernel=False, include_edge=(x_key == "ninner")),
        )

    ax.set_xlabel("ninner" if x_key == "ninner" else "block edge length")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_xlim(
        left=min(row_x(r, x_key) for r in rows) * 0.9,
        right=max(row_x(r, x_key) for r in rows) * 1.1,
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=False)


def add_plot_page(pdf, rows, title, y_key, y_label, x_key="edge"):
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_series(ax, rows, y_key, y_label, title, x_key)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_summary_page(pdf, rows, title):
    best = {}
    for row in rows:
        key = series_key(row)
        if key not in best or row["updates_per_second"] > best[key]["updates_per_second"]:
            best[key] = row

    lines = [title, ""]
    for key in sorted(best):
        row = best[key]
        lines.append(
            f"{series_label(row, include_kernel=True)}: {row['updates_per_second']:.3e} updates/s "
            f"at edge={row_edge(row)} ninner={row['ninner']} blocks={row_blocks(row)}"
        )
    add_text_page(pdf, title, lines)


def main():
    args = parse_args()
    if args.analysis_mode == "full":
        if args.edge_values == "8,32,128":
            args.edge_values = "8,16,32,64,128"
        if args.niter_values == "1,16,128":
            args.niter_values = "1,4,16,64,128"
        if not args.stencil_shapes:
            args.stencil_shapes = "point,x3,y3,z3"
    elif args.analysis_mode == "ninner":
        if not args.ninner_values:
            args.ninner_values = "64,128,256,384,512,640,768,896,1024,1152,1280,1536,1792,2048,3072,4096,8192,16384"
        if args.edge_values == "8,32,128":
            args.edge_values = "32"
    if not args.stencil_shapes:
        args.stencil_shapes = f"{args.stencil_x}/{args.stencil_y}/{args.stencil_z}"

    binary = str(Path(args.binary).resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loops = [loop.strip() for loop in args.loops.split(",") if loop.strip()]
    edge_values = parse_csv_ints(args.edge_values)
    ninner_values = parse_csv_ints(args.ninner_values) if args.ninner_values else []
    stencil_shapes = parse_stencil_shapes(args.stencil_shapes)
    x_key = "ninner" if ninner_values else "edge"
    cases_csv = output_dir / "cases.csv"
    results_csv = output_dir / "results.csv"
    pdf_path = output_dir / "summary.pdf"

    write_cases_csv(cases_csv, loops, edge_values, ninner_values, stencil_shapes, args)
    run_binary(binary, str(cases_csv), str(results_csv))
    rows = numericize_rows(read_results_csv(results_csv))
    meta = collect_metadata(binary)
    kernel_labels = sorted({row_kernel(row) for row in rows}) or ["unknown"]
    kernel_summary = ", ".join(kernel_labels)

    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            f"{args.title} ({kernel_summary})",
            [
                "Sweep",
                f"- analysis mode: {args.analysis_mode}",
                f"- loops: {args.loops}",
                f"- access modes: direct, hoisted",
                f"- edge values: {args.edge_values}",
                f"- niter values: {args.niter_values}",
                f"- stencil shapes: {args.stencil_shapes}",
                f"- target total cells: {args.target_total_cells}",
                f"- ninner values: {args.ninner_values}" if ninner_values else "- ninner = edge^2",
                f"- warmup: {args.warmup}",
                f"- repeats: {args.repeats}",
                "",
                "Environment",
                f"- platform: {meta.get('platform', 'unknown')}",
                f"- cpu: {meta.get('cpu', 'unknown')}",
                f"- uname: {meta.get('uname', 'unknown')}",
                f"- python: {meta.get('python', 'unknown')}",
                f"- compiler: {meta.get('compiler', 'unknown')}",
                f"- compiler id: {meta.get('compiler_id', 'unknown')}",
                f"- compiler version: {meta.get('compiler_version', 'unknown')}",
                f"- compiler text: {meta.get('compiler_version_text', 'unknown')}",
                f"- kokkos backends: {meta.get('kokkos_backends', 'unknown')}",
                "",
                "Compiler flags",
                meta.get("cxx_flags", "unknown"),
                "",
                f"cases csv: {cases_csv}",
                f"results csv: {results_csv}",
            ],
        )

        rows_by_kernel = defaultdict(list)
        for row in rows:
            rows_by_kernel[row_kernel(row)].append(row)

        for kernel_label in sorted(rows_by_kernel):
            kernel_rows = rows_by_kernel[kernel_label]
            add_plot_page(
                pdf,
                kernel_rows,
                f"{args.title}: throughput ({kernel_label})",
                "updates_per_second",
                "updates / second",
                x_key,
            )
            add_summary_page(pdf, kernel_rows, f"{args.title}: best results ({kernel_label})")
            if all("touched_cells_per_second" in row for row in kernel_rows):
                add_plot_page(
                    pdf,
                    kernel_rows,
                    f"{args.title}: touched-cell rate ({kernel_label})",
                    "touched_cells_per_second",
                    "touched cells / second",
                    x_key,
                )

    print(f"Wrote {cases_csv}")
    print(f"Wrote {results_csv}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
