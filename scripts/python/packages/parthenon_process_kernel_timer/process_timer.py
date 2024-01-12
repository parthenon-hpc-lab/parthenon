# =========================================================================================
# (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

"""
Process the text profiling data spit out by the kokkos simple kernel timer, arranging things
in a convenient way, assuming all the region/kernel names are auto-generated by parthenon
"""

import sys


class region:
    def __init__(self, fl, line, data):
        self.file = fl
        self.line = line
        self.data = data


class func:
    def __init__(self, name):
        self.name = name
        self.file = ""
        self.reg = {}
        self.reg_type = {}
        self.total_time = 0.0
        self.total_pct = 0.0
        self.total_kernel_time = 0.0

    def add_region(self, fl, line, data):
        if self.file == "":
            self.file = fl
        else:
            if fl != self.file:
                print("Duplicate function name found")
                sys.exit(1)
        dsplit = data.split()
        self.reg_type[line] = dsplit[0][1:-1]
        self.reg[line] = [float(dsplit[1]), float(dsplit[5].rstrip())]
        if self.reg_type[line] != "REGION":
            self.total_kernel_time += self.reg[line][0]

    def compute_total_cost(self):
        if len(self.reg) == 1:
            for key in self.reg.keys():
                if self.reg_type[key] != "REGION":
                    self.name += "   ***WARNING: no function-level profiling***"
                self.total_time = self.reg[key][0]
                self.total_pct = self.reg[key][1]
            return
        # sort by line number
        keys = list(self.reg.keys())
        key_int = [int(k) for k in keys]
        key_int.sort()
        keys = [str(k) for k in key_int]
        if self.reg_type[keys[0]] == "REGION":
            # assume this is the time for the function
            self.total_time = self.reg[keys[0]][0]
            self.total_pct = self.reg[keys[0]][1]
        else:
            # assume there are just kernel timers
            self.name += "   ***WARNING: no function-level profiling***"
            wall = 0.0
            for key, val in self.reg.items():
                if self.reg_type[key] != "REGION":
                    self.total_time += self.reg[key][0]
                    wall = self.reg[key][0] / self.reg[key][1]
            self.total_pct = self.total_time / wall

    def print_region(self):
        print(
            self.name, "time: " + str(self.total_time), "%wall: " + str(self.total_pct)
        )
        for key, val in self.reg.items():
            if val[0] != self.total_time:
                print(
                    "  ",
                    "type: " + self.reg_type[key],
                    "line: " + key,
                    "selftime: " + str(val[0]),
                    "%func: " + str(100 * val[0] / self.total_time),
                )
        if self.total_time > 0:
            print(
                "  ",
                "Kernel summary: Total kernel time: " + str(self.total_kernel_time),
                "  % Time in kernels: ",
                str(100 * self.total_kernel_time / self.total_time),
            )
        else:
            print("  ", "Apparently this function took zero time to execute???")


def parse_name(s):
    if s[0:6] == "Kokkos":
        f = s.rstrip()
        line = "na"
        fl = "na"
        label = f
    else:
        words = s.split("::")
        if words[0] == "kokkos_abstraction.hpp":
            return "", "", "", "", True
        # make sure it follows the filename::line_number::name convection
        if ".cpp" in s or ".hpp" in s:
            # now strip away the file and line
            fl = s[: s.find(":")]
            f = s[s.find(":") + 2 :]
            line = f[: f.find(":")]
            f = f[f.find(":") + 2 :]
            label = (fl + "::" + f).rstrip()
        else:
            print("nonconforming entry", s)
            sys.exit(1)
    return f, line, fl, label, False


def main(prof):
    funcs = {}
    with open(prof) as fp:
        raw = fp.readlines()
    if raw[0].rstrip() != "Regions:":
        print(
            prof
            + " does not appear to be a profile from the Kokkos simple kernel timer"
        )
        print(raw[0])
        sys.exit(1)
    cline = 2
    in_regions = True
    # process regions/kernels
    while raw[cline].rstrip() != "":
        sraw = raw[cline][2::]
        f, line, fl, label, skip = parse_name(sraw)
        if skip:
            cline += 2
            continue
        if label not in funcs.keys():
            funcs[label] = func(label)
        funcs[label].add_region(fl, line, raw[cline + 1].rstrip())
        cline += 2
        if raw[cline].rstrip() == "" and in_regions:
            cline += 4
            in_regions = False

    for key in funcs.keys():
        funcs[key].compute_total_cost()

    for key in sorted(funcs, key=lambda name: funcs[name].total_time, reverse=True):
        funcs[key].print_region()
        print()


if __name__ == "__main__":
    main(sys.argv[1])
