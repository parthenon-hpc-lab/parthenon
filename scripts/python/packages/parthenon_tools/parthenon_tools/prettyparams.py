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
import sys
import textwrap
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
        decorations = 2 + 3 * (ncols - 1) + 2  # borders & separators
        candidate = term_cols - sum(fixed_widths) - decorations
        w4 = max(24, candidate)
    return fixed_widths + [w4]

def wrap_desc(text, width):
    text = (text or "").strip()
    lines = textwrap.wrap(
        text, width=width, break_long_words=True, break_on_hyphens=False, drop_whitespace=True
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

def run_plain(rows, desc_width=None):
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
    print(mid)
    for row in rows[1:]:
        print_row(row, widths)
        print(sep)

# ---------- Curses TUI mode (collapsible groups by column 0) ----------

def build_groups(rows):
    """
    Returns OrderedDict[str, list[rows]] preserving first appearance of group (col0).
    Assumes rows include header as rows[0]; groups are built from rows[1:].
    """
    groups = OrderedDict()
    for r in rows[1:]:
        key = str(r[0] or "").strip()
        groups.setdefault(key, []).append(r)
    return groups

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

def rebuild_display_buffer(stdscr, header, groups, collapsed, widths):
    """
    Build a list of tuples: (rendered_string, meta)
    meta = dict(kind='header'|'group'|'row', group=group_key)
    """
    h, w = stdscr.getmaxyx()
    # Help line and header consume 2 lines
    help_line = " q:quit  ↑/↓ or j/k:move  PgUp/PgDn:scroll  TAB/ENTER:toggle block  a:toggle all "
    help_line = help_line[:max(0, w-1)]
    display = [(help_line, {"kind": "help"})]

    # Header line
    header_line = f"{START_BORDER}{SEPARATOR_BETWEEN.join(hc.ljust(widths[i]) for i, hc in enumerate(header))}{END_BORDER}"
    display.append((header_line, {"kind": "header"}))

    # Separator (light)
    sep = border_line(widths, char="-", corner="+")
    display.append((sep[:max(0, w-1)], {"kind": "sep"}))

    # Groups
    for gkey, rows in groups.items():
        count = len(rows)
        marker = "[+]" if gkey in collapsed else "[-]"
        label = gkey if gkey else "(blank)"
        gtext = f" {marker} {label}  ({count} row{'s' if count != 1 else ''})"
        display.append((gtext[:max(0, w-1)], {"kind": "group", "group": gkey}))
        if gkey not in collapsed:
            # header separator for table body inside group
            display.append((sep[:max(0, w-1)], {"kind": "sep", "group": gkey}))
            for r in rows:
                for phys in row_to_wrapped_lines(r, widths):
                    display.append((phys[:max(0, w-1)], {"kind": "row", "group": gkey}))
            # trailing separator between groups
            display.append((sep[:max(0, w-1)], {"kind": "sep", "group": gkey}))
    return display

def run_curses(rows, desc_width=None):
    import curses

    rows = strip_empty_rows(rows)
    if not rows:
        print("No rows to display.")
        return
    rows = normalize_rows(rows)
    header = rows[0]
    groups = build_groups(rows)

    def _main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        collapsed = set(groups.keys())  # set of group keys that are collapsed
        top_index = 0                   # first visible line index in buffer
        cursor = 3                      # start after help+header+sep
        while True:
            h, w = stdscr.getmaxyx()
            widths = compute_widths([header] + [r for rs in groups.values() for r in rs],
                                    desc_width=desc_width, term_cols=w)
            buf = rebuild_display_buffer(stdscr, header, groups, collapsed, widths)

            # Clamp indices
            max_idx = max(0, len(buf) - 1)
            cursor = max(0, min(cursor, max_idx))
            # Ensure cursor is visible within window (leaving one line for status)
            view_h = max(1, h - 1)
            if cursor < top_index:
                top_index = cursor
            elif cursor >= top_index + view_h:
                top_index = cursor - view_h + 1

            stdscr.erase()
            # Draw visible window
            for i in range(view_h):
                bi = top_index + i
                if bi >= len(buf): break
                line, meta = buf[bi]
                if bi == cursor:
                    stdscr.addnstr(i, 0, line, w - 1, curses.A_REVERSE)
                else:
                    # Slight styling for group lines
                    attr = curses.A_BOLD if meta.get("kind") == "group" else curses.A_NORMAL
                    stdscr.addnstr(i, 0, line, w - 1, attr)

            # Status line: show current group and hint
            cur_meta = buf[cursor][1] if buf else {}
            status = ""
            if cur_meta.get("kind") in ("group", "row", "sep"):
                g = cur_meta.get("group") or "(blank)"
                folded = "hidden" if (cur_meta.get("group") in collapsed) else "visible"
                if cur_meta.get("kind") == "group":
                    status = f" Group: {g}  [{folded}] — press TAB or ENTER to toggle "
                else:
                    status = f" Group: {g}  [{folded}] — press TAB or ENTER on the group line to toggle "
            stdscr.addnstr(h - 1, 0, status.ljust(w - 1), w - 1, curses.A_DIM)

            ch = stdscr.getch()
            if ch in (ord('q'), ord('Q')):
                break
            elif ch in (curses.KEY_UP, ord('k')):
                cursor = max(0, cursor - 1)
            elif ch in (curses.KEY_DOWN, ord('j')):
                cursor = min(max_idx, cursor + 1)
            elif ch == curses.KEY_PPAGE:  # Page Up
                cursor = max(0, cursor - (view_h - 1))
            elif ch == curses.KEY_NPAGE:  # Page Down
                cursor = min(max_idx, cursor + (view_h - 1))
            elif ch in (curses.KEY_RESIZE,):
                pass  # loop will recompute widths/buffer
            elif ch in (9, curses.KEY_BTAB, 10, 13):  # Tab or Enter
                # Toggle the group of the current line; if on a group line, use that group
                meta = buf[cursor][1]
                g = meta.get("group")
                if meta.get("kind") == "group" and g is not None:
                    if g in collapsed: collapsed.remove(g)
                    else: collapsed.add(g)
                else:
                    # Find nearest group line above
                    gi = cursor
                    while gi >= 0 and buf[gi][1].get("kind") != "group":
                        gi -= 1
                    if gi >= 0:
                        g = buf[gi][1].get("group")
                        if g in collapsed: collapsed.remove(g)
                        else: collapsed.add(g)
            elif ch in (ord('a'), ord('A')):
                if len(collapsed) < len(groups):
                    collapsed = set(groups.keys())   # collapse all
                else:
                    collapsed.clear()                # expand all
            # else: ignore other keys

    import curses
    curses.wrapper(_main)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Pretty-print parthenon params output, or browse it interactively with collapsible groups."
    )
    ap.add_argument("paramsfile", nargs="?", help="CSV files to load (default: stdin).")
    ap.add_argument("--desc-width", type=int, default=None,
                    help="Set width of the description column (both modes).")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="Launch interactive UI where params are grouped by block and can be hidden.")
    args = ap.parse_args()

    rows = read_csv(args.paramsfile)
    rows = strip_empty_rows(rows)
    if not rows:
        return

    if args.interactive:
        run_curses(rows, desc_width=args.desc_width)
    else:
        run_plain(rows, desc_width=args.desc_width)

if __name__ == "__main__":
    main()
