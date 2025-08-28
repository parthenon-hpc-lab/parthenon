#!/usr/bin/env python3

# =========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020-2024 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# =========================================================================================
# (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# =========================================================================================

# This script was generated with the help of Generative AI. The first
# draft was built with the help of ChatGPT5.

import argparse
import csv
import shutil
import os, sys
import textwrap
import re
from collections import OrderedDict

# ---------- Shared formatting helpers ----------

SEPARATOR_BETWEEN = " | "
START_BORDER = "| "
END_BORDER = " |"


def strip_empty_rows(rows):
    return [r for r in rows if any((c or "").strip() for c in r)]


def read_csv(path):
    if path is None:  # stdin
        f = sys.stdin
    else:
        f = open(path, newline="", encoding="utf-8")
    try:
        return [row for row in csv.reader(f)]
    finally:
        if path is not None:
            f.close()


def normalize_rows(rows, ncols=5):
    "Adds empty columns to the csv data structure to ensure total number of columns is consistent"
    return [row[:ncols] + [""] * (ncols - len(row)) for row in rows]


def compute_widths(rows, desc_width=None, term_cols=None):
    # 5 columns: block, parameters, type, default, description
    ncols = 5
    rows = normalize_rows(rows, ncols)
    fixed_widths = [max(len(row[i]) for row in rows) for i in range(4)]
    if desc_width:
        w4 = desc_width
    else:
        if term_cols is None:
            term_cols = shutil.get_terminal_size(fallback=(120, 20)).columns
        # borders & separators: start/end + 3 separators between 5 cols
        decorations = 2 + 3 * (ncols - 1) + 2
        candidate = term_cols - sum(fixed_widths) - decorations
        w4 = max(24, candidate)
    return fixed_widths + [w4]


def wrap_desc(text, width):
    text = (text or "").strip()
    lines = textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        drop_whitespace=True,
    )
    return lines or [""]


def border_line(widths, char="-", corner="+"):
    parts = [corner]
    for w in widths:
        parts.append(char * (w + 2))
        parts.append(corner)
    return "".join(parts)


def print_row(cells, widths):
    # Wrap only description (col 4)
    lines_per_col = []
    for i, (cell, w) in enumerate(zip(cells, widths)):
        s = str(cell or "")
        lines_per_col.append(wrap_desc(s, w) if i == 4 else [s])
    height = max(len(l) for l in lines_per_col)
    for i in range(len(lines_per_col)):
        lines_per_col[i] += [""] * (height - len(lines_per_col[i]))
    for row_line in range(height):
        parts = [START_BORDER]
        for j, w in enumerate(widths):
            parts.append(lines_per_col[j][row_line].ljust(w))
            if j < len(widths) - 1:
                parts.append(SEPARATOR_BETWEEN)
        parts.append(END_BORDER)
        print("".join(parts))


# ---------- Plain ASCII table mode (default) ----------


def run_plain(rows, desc_width=None, no_header_sep=False):
    rows = strip_empty_rows(rows)
    if not rows:
        return
    rows = normalize_rows(rows)
    widths = compute_widths(rows, desc_width=desc_width)

    top = border_line(widths, char="-", corner="+")
    mid = border_line(widths, char="=", corner="+")
    sep = border_line(widths, char="-", corner="+")

    print(top)
    print_row(rows[0], widths)
    print(mid if not no_header_sep else sep)
    for row in rows[1:]:
        print_row(row, widths)
        print(sep)


# ---------- Curses TUI mode (collapsible blocks + search + highlight) ----------


def reattach_tty_for_curses():
    """
    Ensure sys.stdin/sys.stdout are connected to the terminal before initializing curses.
    """
    try:
        # If we already have a tty, nothing to do
        if sys.stdin.isatty() and sys.stdout.isatty():
            return

        # Try to open /dev/tty
        tty_in = open("/dev/tty", "rb", buffering=0)
        tty_out = open("/dev/tty", "wb", buffering=0)
        os.dup2(tty_in.fileno(), sys.stdin.fileno())
        os.dup2(tty_out.fileno(), sys.stdout.fileno())
        return
    except Exception:
        sys.stderr.write(
            "Error: No TTY available for UI mode. "
            "If you're piping input, run like:\n"
            "   cat file.csv | python prettyparams.py --interactive\n"
            "Or pass a filename directly:\n"
            "   python prettyparams.py --interactive file.csv\n"
        )
        sys.exit(1)


def build_blocks(rows):
    """
    Returns OrderedDict[str, list[rows]] preserving first appearance of block (col0).
    Assumes rows include header as rows[0]; blocks are built from rows[1:].
    """
    blocks = OrderedDict()
    for r in rows[1:]:
        key = str(r[0] or "").strip()
        blocks.setdefault(key, []).append(r)
    return blocks


def row_to_wrapped_lines(row, widths):
    # Return list of physical lines (strings) for a single data row based on wrapping of description
    cells = normalize_rows([row])[0]
    desc_lines = wrap_desc(cells[4], widths[4])
    out = []
    for i, dline in enumerate(desc_lines):
        parts = []
        for col_idx, w in enumerate(widths):
            if col_idx == 4:
                txt = dline
            else:
                txt = cells[col_idx] if i == 0 else ""
            parts.append(txt.ljust(w))
        out.append(f"{START_BORDER}{SEPARATOR_BETWEEN.join(parts)}{END_BORDER}")
    return out


def filter_blocks(blocks, pattern):
    """Filter rows in each block by regex pattern across all columns. Return new blocks dict."""
    if not pattern:
        return blocks, None
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return None, f"Invalid regex: {e}"

    filtered = OrderedDict()
    for gkey, rows in blocks.items():
        kept = []
        gmatch = rx.search(gkey or "")
        for r in rows:
            if gmatch or any(rx.search(str(c or "")) for c in r):
                kept.append(r)
        if kept:
            filtered[gkey] = kept
    return filtered, None


def rebuild_display_buffer(stdscr, header, blocks, collapsed, widths, active_filter):
    """
    Build a list of tuples: (rendered_string, meta)
    meta = dict(kind='header'|'block'|'row'|'sep'|'help', block=block_key)
    While a filter is active, blocks are shown expanded regardless of 'collapsed'.
    """
    h, w = stdscr.getmaxyx()
    help_line = " q:quit  ↑/↓ or j/k:move  PgUp/PgDn:scroll  TAB/ENTER:toggle block  a:toggle all  /:search  c:clear "
    help_line = help_line[: max(0, w - 1)]
    display = [(help_line, {"kind": "help"})]

    # Header line
    header_line = f"{START_BORDER}{SEPARATOR_BETWEEN.join(hc.ljust(widths[i]) for i, hc in enumerate(header))}{END_BORDER}"
    display.append((header_line, {"kind": "header"}))

    # Separator
    sep = border_line(widths, char="-", corner="+")
    display.append((sep[: max(0, w - 1)], {"kind": "sep"}))

    filtering = active_filter is not None and active_filter != ""
    for gkey, rows in blocks.items():
        count = len(rows)
        expanded = True if filtering else (gkey not in collapsed)
        marker = "[-]" if expanded else "[+]"
        label = gkey if gkey else "(blank)"
        suffix = f" (filter: {active_filter})" if filtering else ""
        gtext = f" {marker} {label}  ({count} row{'s' if count != 1 else ''}){suffix if filtering else ''}"
        display.append((gtext[: max(0, w - 1)], {"kind": "block", "block": gkey}))
        if expanded:
            display.append((sep[: max(0, w - 1)], {"kind": "sep", "block": gkey}))
            for r in rows:
                for phys in row_to_wrapped_lines(r, widths):
                    display.append(
                        (phys[: max(0, w - 1)], {"kind": "row", "block": gkey})
                    )
            display.append((sep[: max(0, w - 1)], {"kind": "sep", "block": gkey}))
    return display


def prompt_input(stdscr, prompt):
    """Simple in-line input at bottom; returns string or None if cancelled (ESC)."""
    import curses

    h, w = stdscr.getmaxyx()
    y = h - 1
    stdscr.move(y, 0)
    stdscr.clrtoeol()
    stdscr.addnstr(y, 0, prompt, w - 1, curses.A_BOLD)
    s = ""
    curses.curs_set(1)
    while True:
        ch = stdscr.getch()
        if ch in (10, 13):  # Enter
            curses.curs_set(0)
            return s
        if ch == 27:  # ESC
            curses.curs_set(0)
            return None
        if ch in (curses.KEY_BACKSPACE, 127, 8):
            if s:
                s = s[:-1]
                stdscr.move(y, len(prompt))
                stdscr.clrtoeol()
                stdscr.addnstr(y, len(prompt), s, w - 1)
        elif 0 <= ch < 256:
            s += chr(ch)
            stdscr.addnstr(y, len(prompt), s, w - 1)


# --- NEW: highlighting helpers ---


def compile_filter_regex(active_filter):
    """Return compiled regex or None if no/invalid pattern."""
    if not active_filter:
        return None
    try:
        return re.compile(active_filter, re.IGNORECASE)
    except re.error:
        return None  # invalid patterns already surfaced elsewhere


def find_spans(text, rx):
    """Return non-overlapping (start, end) spans for matches of rx in text."""
    if rx is None or not text:
        return []
    spans = []
    for m in rx.finditer(text):
        a, b = m.span()
        if a < b:
            if spans and a <= spans[-1][1]:
                # merge overlapping/adjacent spans
                spans[-1] = (spans[-1][0], max(spans[-1][1], b))
            else:
                spans.append((a, b))
    return spans


def addstr_with_highlight(
    stdscr, y, x, text, width, is_selected, spans, base_bold=False
):
    """Draw text with highlighted spans; respects selection inverse."""
    import curses

    if width <= 0:
        return
    maxlen = max(0, width - x - 1)
    text = text[:maxlen]
    idx = 0
    ptr = 0
    # choose base attribute
    base_attr = curses.A_REVERSE if is_selected else curses.A_NORMAL
    if base_bold:
        base_attr |= curses.A_BOLD
    hi_attr = curses.A_BOLD | curses.A_STANDOUT
    if is_selected:
        hi_attr |= curses.A_REVERSE
    for a, b in spans:
        if a > len(text):
            break
        # plain region
        if ptr < a:
            stdscr.addnstr(y, x + idx, text[ptr:a], maxlen - idx, base_attr)
            idx += a - ptr
            ptr = a
        # highlight region
        hb = min(b, len(text))
        if ptr < hb:
            stdscr.addnstr(y, x + idx, text[ptr:hb], maxlen - idx, hi_attr)
            idx += hb - ptr
            ptr = hb
    # tail
    if ptr < len(text):
        stdscr.addnstr(y, x + idx, text[ptr:], maxlen - idx, base_attr)


def run_curses(rows, desc_width=None):
    import curses

    rows = strip_empty_rows(rows)
    if not rows:
        print("No rows to display.")
        return
    rows = normalize_rows(rows)
    header = rows[0]
    base_blocks = build_blocks(rows)

    def _main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)

        collapsed = set(base_blocks.keys())  # start with all blocks collapsed
        active_filter = ""  # regex string (empty = no filter)
        top_index = 0  # first visible line index in buffer
        cursor = 3  # initial cursor line (after help+header+sep)

        def current_blocks():
            if active_filter:
                fg, err = filter_blocks(base_blocks, active_filter)
                if err:
                    return None, err
                return fg, None
            return base_blocks, None

        while True:
            h, w = stdscr.getmaxyx()
            blocks, ferr = current_blocks()
            if blocks is None:  # invalid regex; show message and keep last valid state
                blocks = base_blocks
            # widths from all visible rows
            all_rows = [header] + [r for rs in blocks.values() for r in rs]
            widths = compute_widths(all_rows, desc_width=desc_width, term_cols=w)
            buf = rebuild_display_buffer(
                stdscr, header, blocks, collapsed, widths, active_filter
            )

            # Clamp indices
            max_idx = max(0, len(buf) - 1)
            cursor = max(0, min(cursor, max_idx))
            view_h = max(1, h - 1)
            if cursor < top_index:
                top_index = cursor
            elif cursor >= top_index + view_h:
                top_index = cursor - view_h + 1

            stdscr.erase()

            # Compile regex for highlighting
            rx = compile_filter_regex(active_filter)

            # Draw visible window with highlighting
            for i in range(view_h):
                bi = top_index + i
                if bi >= len(buf):
                    break
                line, meta = buf[bi]
                is_sel = bi == cursor
                kind = meta.get("kind")
                # Header & block lines get bold base for readability
                base_bold = kind in ("block", "header")
                # Compute highlight spans for this line if filter is active
                spans = find_spans(line, rx)
                addstr_with_highlight(
                    stdscr, i, 0, line, w, is_sel, spans, base_bold=base_bold
                )

            # Status line: errors / filter state
            status = ""
            if ferr:
                status = f" {ferr}  (press / to edit, c to clear)"
            elif active_filter:
                status = f" filter: /{active_filter}/i — TAB/ENTER toggle blocks; 'c' clears "
            stdscr.addnstr(h - 1, 0, status.ljust(w - 1), w - 1)

            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q")):
                break
            elif ch in (curses.KEY_UP, ord("k")):
                cursor = max(0, cursor - 1)
            elif ch in (curses.KEY_DOWN, ord("j")):
                cursor = min(max_idx, cursor + 1)
            elif ch == curses.KEY_PPAGE:
                cursor = max(0, cursor - (view_h - 1))
            elif ch == curses.KEY_NPAGE:
                cursor = min(max_idx, cursor + (view_h - 1))
            elif ch in (curses.KEY_RESIZE,):
                pass
            elif ch in (ord("a"), ord("A")):  # toggle all
                if len(collapsed) < len(base_blocks):
                    collapsed = set(base_blocks.keys())  # collapse all
                else:
                    collapsed.clear()  # expand all
            elif ch in (
                9,
                curses.KEY_BTAB,
                10,
                13,
            ):  # Tab or Enter toggles nearest block
                meta = buf[cursor][1] if buf else {}
                g = meta.get("block")
                if meta.get("kind") == "block" and g is not None:
                    if g in collapsed:
                        collapsed.remove(g)
                    else:
                        collapsed.add(g)
                else:
                    gi = cursor
                    while gi >= 0 and buf[gi][1].get("kind") != "block":
                        gi -= 1
                    if gi >= 0:
                        g = buf[gi][1].get("block")
                        if g in collapsed:
                            collapsed.remove(g)
                        else:
                            collapsed.add(g)
            elif ch == ord("/"):  # search / filter (regex)
                s = prompt_input(stdscr, " / (regex, ESC=cancel, empty=clear): ")
                if s is None:
                    pass
                else:
                    active_filter = s
            elif ch in (ord("c"), ord("C")):  # clear filter
                active_filter = ""
            # else: ignore other keys

    import curses

    curses.wrapper(_main)


# ---------- CLI ----------


def main():
    ap = argparse.ArgumentParser(
        description="Pretty-print a parthenon params CSV file as an ASCII table, or browse it interactively."
    )
    ap.add_argument("csv", nargs="?", help="CSV file (default: stdin)")
    ap.add_argument(
        "-w",
        "--desc-width",
        type=int,
        default=None,
        help="Set width of the description column (both modes).",
    )
    ap.add_argument(
        "--no-header-sep",
        action="store_true",
        help="(ascii mode) Do not print a heavy separator under the header row.",
    )
    ap.add_argument(
        "-i", "--interactive", action="store_true", help="Launch interactive TUI"
    )
    args = ap.parse_args()

    rows = read_csv(args.csv)
    rows = strip_empty_rows(rows)
    if not rows:
        return

    if args.interactive:
        reattach_tty_for_curses()
        run_curses(rows, desc_width=args.desc_width)
    else:
        run_plain(rows, desc_width=args.desc_width, no_header_sep=args.no_header_sep)


if __name__ == "__main__":
    main()
