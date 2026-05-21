#!/usr/bin/env python3

import argparse
import csv
import os
import sys
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "parthenon-loop-mpl")
)
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def maybe_add_pypdf_path():
    for candidate in ("/tmp/pypdf-merge",):
        if Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)


maybe_add_pypdf_path()
from pypdf import PdfReader, PdfWriter


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lookup_updates(rows, experiment, variant, ni=None, inner_chunk_length=None):
    for row in rows:
        if row["experiment"] != experiment or row["variant"] != variant:
            continue
        if ni is not None and int(row["ni"]) != ni:
            continue
        if (
            inner_chunk_length is not None
            and int(row["inner_chunk_length"]) != inner_chunk_length
        ):
            continue
        return float(row["updates_per_second"])
    raise KeyError((experiment, variant, ni, inner_chunk_length))


def build_summary(intel_rows, m4_rows, verify_rows):
    intel_stencil_ratio_8 = lookup_updates(
        intel_rows, "stencil_ni_sweep", "cpu_coalesced_outer_var", ni=8
    ) / lookup_updates(intel_rows, "stencil_ni_sweep", "cpu_rowvar_simd", ni=8)
    intel_stencil_ratio_32 = lookup_updates(
        intel_rows, "stencil_ni_sweep", "cpu_coalesced_outer_var", ni=32
    ) / lookup_updates(intel_rows, "stencil_ni_sweep", "cpu_rowvar_simd", ni=32)

    intel_chunk_small = lookup_updates(
        intel_rows,
        "stencil_chunk_sweep",
        "cpu_hierarchical",
        ni=32,
        inner_chunk_length=8,
    )
    intel_chunk_large = lookup_updates(
        intel_rows,
        "stencil_chunk_sweep",
        "cpu_hierarchical",
        ni=32,
        inner_chunk_length=256,
    )

    m4_chunk_small = lookup_updates(
        m4_rows, "stencil_chunk_sweep", "cpu_hierarchical", ni=32, inner_chunk_length=8
    )
    m4_chunk_large = lookup_updates(
        m4_rows,
        "stencil_chunk_sweep",
        "cpu_hierarchical",
        ni=32,
        inner_chunk_length=1024,
    )

    verify_heavy_ratio_32 = lookup_updates(
        verify_rows, "heavy_ni_sweep", "cpu_coalesced_outer_var", ni=32
    ) / lookup_updates(verify_rows, "heavy_ni_sweep", "cpu_simd", ni=32)
    verify_heavy_ratio_128 = lookup_updates(
        verify_rows, "heavy_ni_sweep", "cpu_coalesced_outer_var", ni=128
    ) / lookup_updates(verify_rows, "heavy_ni_sweep", "cpu_simd", ni=128)

    return {
        "intel_stencil_ratio_8": intel_stencil_ratio_8,
        "intel_stencil_ratio_32": intel_stencil_ratio_32,
        "intel_chunk_gain": intel_chunk_large / intel_chunk_small,
        "m4_chunk_gain": m4_chunk_large / m4_chunk_small,
        "verify_heavy_ratio_32": verify_heavy_ratio_32,
        "verify_heavy_ratio_128": verify_heavy_ratio_128,
    }


def wrap_lines(items, width=98):
    lines = []
    for item in items:
        if item == "":
            lines.append("")
        else:
            lines.extend(
                textwrap.wrap(
                    item, width=width, break_long_words=False, break_on_hyphens=False
                )
            )
    return lines


def create_intro_pdf(path, intel_pdf, m4_pdf, summary):
    page1 = [
        "CPU Loop Benchmark Interpretation",
        "",
        "Scope",
        f"- Included reports: Intel Xeon analysis from {intel_pdf}",
        f"- Included reports: current M4 rerun from {m4_pdf}",
        "- Supporting verification data came from verify.csv in this repo root.",
        "",
        "Interpretation",
        "- The old CPU conclusion was likely partly misattributed. The transformation that made SIMD easier for the compiler also improved memory locality, chunk amortization, and streaming behavior.",
        "- Across both the Intel Xeon and the Apple M4 runs, stencil performance is dominated more by memory-system behavior and loop organization than by a simple yes/no SIMD story.",
        f"- On the Intel default stencil ni sweep, cpu_coalesced_outer_var stays about {summary['intel_stencil_ratio_8']:.1f}x faster than cpu_rowvar_simd at ni=8 and about {summary['intel_stencil_ratio_32']:.1f}x faster at ni=32, so the multi-row raw-span pattern is still strong.",
        f"- In the fixed-ni stencil chunk sweep, cpu_hierarchical gains about {summary['intel_chunk_gain']:.1f}x on Intel and about {summary['m4_chunk_gain']:.1f}x on M4 when chunk size grows from a very small value to a large one. That is much more consistent with locality/prefetch/amortization effects than with vector width alone.",
        "- The strongest performance jumps appear when chunk sizes become large enough to amortize per-chunk and per-variable overhead, not at a single universal SIMD threshold.",
    ]

    page2 = [
        "Verification Readout",
        "",
        "- Verify mode forces chunk=ni for the CPU raw-span variants during the ni sweep, making the raw-span and plain CPU SIMD rows much closer in loop structure.",
        f"- In verify.csv for the heavy kernel, cpu_coalesced_outer_var / cpu_simd is {summary['verify_heavy_ratio_32']:.3f} at ni=32 and {summary['verify_heavy_ratio_128']:.3f} at ni=128.",
        "- That remaining gap is only about 4-5 percent, which is small enough to attribute to loop scaffolding and code-generation differences rather than to a different memory-access regime.",
        "- This supports the view that the CPU debate is largely settled: the major wins come from memory-friendly loop structure, while the residual gap between otherwise similar CPU variants is comparatively minor.",
        "",
        "Practical Takeaway",
        "- Multi-row raw-span remains a strong CPU pattern, but the margin over alternatives is machine-dependent and narrows once chunking and streaming become favorable for the competing loop orders.",
        "- The benchmark results do not suggest a fundamental bug in the CPU implementations.",
        "- A better historical interpretation is: SIMD-enabling transformations helped, but much of the observed gain likely came from improved use of the memory subsystem.",
    ]

    with PdfPages(path) as pdf:
        for title, items in (
            ("CPU Interpretation", page1),
            ("Verification Summary", page2),
        ):
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            ax.text(
                0.05,
                0.97,
                "\n".join(wrap_lines(items)),
                va="top",
                ha="left",
                family="monospace",
                fontsize=11,
            )
            ax.set_title(title, loc="left", fontsize=14, pad=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def merge_pdfs(output_path, *inputs):
    writer = PdfWriter()
    for input_path in inputs:
        reader = PdfReader(str(input_path))
        for page in reader.pages:
            writer.add_page(page)
    with open(output_path, "wb") as handle:
        writer.write(handle)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a combined CPU interpretation PDF."
    )
    parser.add_argument("--intel-csv", required=True)
    parser.add_argument("--intel-pdf", required=True)
    parser.add_argument("--m4-csv", required=True)
    parser.add_argument("--m4-pdf", required=True)
    parser.add_argument("--verify-csv", required=True)
    parser.add_argument("--output-pdf", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    intel_rows = read_rows(args.intel_csv)
    m4_rows = read_rows(args.m4_csv)
    verify_rows = read_rows(args.verify_csv)
    summary = build_summary(intel_rows, m4_rows, verify_rows)

    output_pdf = Path(args.output_pdf)
    intro_pdf = output_pdf.with_name(output_pdf.stem + "-intro.pdf")

    create_intro_pdf(intro_pdf, args.intel_pdf, args.m4_pdf, summary)
    merge_pdfs(output_pdf, intro_pdf, args.intel_pdf, args.m4_pdf)

    print(f"Wrote {intro_pdf}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
