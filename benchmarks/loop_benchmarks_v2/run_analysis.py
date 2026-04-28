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
]

LOOP_STYLES = {
    "cpu_flat_ghosts": ("#264653", "o"),
    "cpu_boiv_contiguous": ("#6d597a", "s"),
    "cpu_bovi_contiguous": ("#2a9d8f", "^"),
    "cpu_bvoi_contiguous": ("#b56576", "D"),
    "cpu_boiv_logical": ("#457b9d", "<"),
    "cpu_bovi_logical": ("#f4a261", ">"),
    "cpu_bvoi_logical": ("#8ab17d", "P"),
    "kokkos_boiv_flat": ("#1d3557", "o"),
    "kokkos_bovi_team_contiguous": ("#7f5539", "s"),
    "kokkos_bovi_team_logical": ("#bc6c25", "^"),
}

PAGE_W = 612.0
PAGE_H = 792.0
MARGIN = 42.0


def parse_csv_ints(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the v2 loop benchmark analysis.")
    parser.add_argument("--binary", default="build-make/benchmarks/loop_benchmarks_v2/loop-benchmarks-v2")
    parser.add_argument("--output-dir", default="reports/loop_benchmarks_v2")
    parser.add_argument("--title", default="Parthenon Loop Benchmark v2")
    parser.add_argument("--loops", default=",".join(DEFAULT_LOOPS))
    parser.add_argument("--edge-values", default="4,8,16,32,64")
    parser.add_argument("--target-total-cells", type=int, default=8_388_608)
    parser.add_argument("--nvars", type=int, default=16)
    parser.add_argument("--nghost", type=int, default=2)
    parser.add_argument("--niter", type=int, default=4)
    parser.add_argument("--stencil-x", type=int, default=1)
    parser.add_argument("--stencil-y", type=int, default=1)
    parser.add_argument("--stencil-z", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
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


def write_cases_csv(path, loops, edge_values, args):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "loop",
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
            for loop in loops:
                writer.writerow(
                    {
                        "loop": loop,
                        "backend": args.backend,
                        "nblocks": 0,
                        "target_cells": args.target_total_cells,
                        "nvars": args.nvars,
                        "nz": edge,
                        "ny": edge,
                        "nx": edge,
                        "nghost": args.nghost,
                        "ninner": edge * edge,
                        "niter": args.niter,
                        "stencil_x": args.stencil_x,
                        "stencil_y": args.stencil_y,
                        "stencil_z": args.stencil_z,
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "vars_per_block": "",
                    }
                )


def run_binary(binary, cases_csv, results_csv):
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
    return row.get("nx", row.get("nx_interior"))


def row_blocks(row):
    return row.get("nblocks", row.get("blocks"))


def wrap_lines(items, width=86):
    lines = []
    for item in items:
        if item == "":
            lines.append("")
        else:
            lines.extend(
                textwrap.wrap(item, width=width, break_long_words=False, break_on_hyphens=False)
            )
    return lines


def pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PdfBuilder:
    def __init__(self):
        self.pages = []

    def add_page(self, commands):
        self.pages.append(commands)

    def _obj(self, obj_id, body):
        return f"{obj_id} 0 obj\n{body}\nendobj\n"

    def write(self, path):
        objects = []
        font_id = 1
        objects.append(self._obj(font_id, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

        page_ids = []
        content_ids = []
        next_id = 2
        for _ in self.pages:
            content_ids.append(next_id)
            page_ids.append(next_id + 1)
            next_id += 2

        pages_tree_id = next_id
        catalog_id = next_id + 1

        for page_commands, content_id in zip(self.pages, content_ids):
            stream = page_commands.encode("utf-8")
            content = f"<< /Length {len(stream)} >>\nstream\n".encode("utf-8") + stream + b"\nendstream"
            objects.append(self._obj(content_id, content.decode("latin1")))

        for page_id, content_id in zip(page_ids, content_ids):
            page_body = (
                f"<< /Type /Page /Parent {pages_tree_id} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            )
            objects.append(self._obj(page_id, page_body))

        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        objects.append(self._obj(pages_tree_id, f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"))
        objects.append(self._obj(catalog_id, f"<< /Type /Catalog /Pages {pages_tree_id} 0 R >>"))

        xref_offsets = []
        pdf = ["%PDF-1.4\n"]
        for obj in objects:
            xref_offsets.append(sum(len(part.encode("latin1")) for part in pdf))
            pdf.append(obj)
        xref_start = sum(len(part.encode("latin1")) for part in pdf)
        pdf.append(f"xref\n0 {len(objects)+1}\n")
        pdf.append("0000000000 65535 f \n")
        for offset in xref_offsets:
            pdf.append(f"{offset:010d} 00000 n \n")
        pdf.append(
            f"trailer << /Size {len(objects)+1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"
        )

        with open(path, "wb") as handle:
            handle.write("".join(pdf).encode("latin1"))


def text_page(title, lines):
    commands = []
    commands.append("BT")
    commands.append("/F1 18 Tf")
    commands.append(f"50 {PAGE_H - 45:.1f} Td")
    commands.append(f"({pdf_escape(title)}) Tj")
    commands.append("ET")
    y = PAGE_H - 75
    for line in wrap_lines(lines, width=88):
        if line == "":
            y -= 10
            continue
        commands.append("BT")
        commands.append("/F1 10 Tf")
        commands.append(f"50 {y:.1f} Td")
        commands.append(f"({pdf_escape(line)}) Tj")
        commands.append("ET")
        y -= 12
    return "\n".join(commands)


def line_plot_page(title, rows, value_key, y_label):
    groups = defaultdict(list)
    for row in rows:
        groups[row["loop"]].append(row)

    series = []
    for loop, points in sorted(groups.items()):
        points = sorted(points, key=row_edge)
        xs = [row_edge(row) for row in points]
        ys = [row[value_key] for row in points]
        series.append((loop, xs, ys))

    all_x = sorted({x for _loop, xs, _ys in series for x in xs})
    all_y = [y for _loop, _xs, ys in series for y in ys]
    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    left, right = 80.0, 560.0
    bottom, top = 110.0, 670.0

    def x_map(value):
        if x_max == x_min:
            return left
        return left + (value - x_min) * (right - left) / (x_max - x_min)

    def y_map(value):
        return bottom + (value - y_min) * (top - bottom) / (y_max - y_min)

    cmds = []
    cmds.append("0 0 0 RG")
    cmds.append("0 0 0 rg")
    cmds.append("BT /F1 16 Tf")
    cmds.append(f"50 {PAGE_H - 45:.1f} Td")
    cmds.append(f"({pdf_escape(title)}) Tj")
    cmds.append("ET")
    cmds.append(f"{left} {bottom} m {right} {bottom} l {right} {top} l S")
    cmds.append(f"BT /F1 10 Tf {((left+right)/2)-40:.1f} {70:.1f} Td ({pdf_escape('block edge length')}) Tj ET")
    cmds.append(f"BT /F1 10 Tf 20 {(bottom+top)/2:.1f} Td ({pdf_escape(y_label)}) Tj ET")

    for x in all_x:
        xpos = x_map(x)
        cmds.append(f"0.85 0.85 0.85 RG")
        cmds.append(f"{xpos:.1f} {bottom} m {xpos:.1f} {top} l S")
        cmds.append("0 0 0 RG")
        cmds.append(f"BT /F1 9 Tf {xpos-8:.1f} {bottom-16:.1f} Td ({x}) Tj ET")

    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4.0
        ypos = y_map(value)
        cmds.append("0.85 0.85 0.85 RG")
        cmds.append(f"{left} {ypos:.1f} m {right} {ypos:.1f} l S")
        cmds.append("0 0 0 RG")
        cmds.append(f"BT /F1 9 Tf {left-42:.1f} {ypos-3:.1f} Td ({value:.2e}) Tj ET")

    legend_x = 370.0
    legend_y = 690.0
    for loop, xs, ys in series:
        color, _marker = LOOP_STYLES.get(loop, ("#000000", "o"))
        r = int(color[1:3], 16) / 255.0
        g = int(color[3:5], 16) / 255.0
        b = int(color[5:7], 16) / 255.0
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} RG")
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
        points = list(zip(xs, ys))
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            cmds.append(f"{x_map(x0):.1f} {y_map(y0):.1f} m {x_map(x1):.1f} {y_map(y1):.1f} l S")
        for x0, y0 in points:
            cmds.append(f"{x_map(x0)-2:.1f} {y_map(y0)-2:.1f} 4 4 re f")
        cmds.append("0 0 0 RG")
        cmds.append(f"BT /F1 9 Tf {legend_x:.1f} {legend_y:.1f} Td ({pdf_escape(loop)}) Tj ET")
        legend_y -= 12

    return "\n".join(cmds)


def summary_page(title, rows):
    best = {}
    for row in rows:
        loop = row["loop"]
        current = best.get(loop)
        if current is None or row["updates_per_second"] > current["updates_per_second"]:
            best[loop] = row

    lines = [title, ""]
    for loop in sorted(best):
        row = best[loop]
        lines.append(
            f"{loop}: {row['updates_per_second']:.3e} updates/s at edge={row_edge(row)} blocks={row_blocks(row)}"
        )
    return text_page(title, lines)


def collect_lines_for_metadata(args, meta, cases_csv, results_csv):
    lines = [
        "Sweep",
        f"- loops: {args.loops}",
        f"- edge values: {args.edge_values}",
        f"- target total cells: {args.target_total_cells}",
        f"- ninner = edge^2",
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
    ]
    return lines


def main():
    args = parse_args()
    binary = str(Path(args.binary).resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loops = [loop.strip() for loop in args.loops.split(",") if loop.strip()]
    edge_values = parse_csv_ints(args.edge_values)
    cases_csv = output_dir / "cases.csv"
    results_csv = output_dir / "results.csv"
    pdf_path = output_dir / "summary.pdf"

    write_cases_csv(cases_csv, loops, edge_values, args)
    run_binary(binary, str(cases_csv), str(results_csv))
    rows = numericize_rows(read_results_csv(results_csv))
    meta = collect_metadata(binary)

    pdf = PdfBuilder()
    pdf.add_page(text_page(args.title, collect_lines_for_metadata(args, meta, cases_csv, results_csv)))
    pdf.add_page(line_plot_page(f"{args.title}: throughput", rows, "updates_per_second", "updates / second"))
    pdf.add_page(
        line_plot_page(
            f"{args.title}: throughput per block",
            rows,
            "updates_per_second",
            "updates / second / block",
        )
    )
    pdf.add_page(summary_page(f"{args.title}: best results", rows))
    pdf.write(pdf_path)

    print(f"Wrote {cases_csv}")
    print(f"Wrote {results_csv}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
