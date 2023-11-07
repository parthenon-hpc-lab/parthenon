import sys
import re
import matplotlib.pyplot as plt
import itertools
from collections import OrderedDict
from argparse import ArgumentParser

parser = ArgumentParser(prog="plot_trace", description="Plot parthenon tracer output")

parser.add_argument(
    "--start",
    dest="step_start",
    type=int,
    default=-1,
    help="First step to include",
)

parser.add_argument(
    "--stop",
    dest="step_stop",
    type=int,
    default=-1,
    help="Final step to include (inclusive)",
)

parser.add_argument("files", type=str, nargs="+", help="trace files to plot")


class Region:
    def __init__(self):
        self.start = []
        self.duration = []

    def add_samples(self, line):
        words = line.split()
        start = float(words[0])
        stop = float(words[1])
        self.start.append(start)
        self.duration.append(stop - start)

    def trim(self, tstart, tstop):
        istart = 0
        istop = 0
        set_start = False
        set_stop = False
        for i in range(len(self.start)):
            if not set_start and self.start[i] > tstart:
                istart = i
                set_start = True
            if not set_stop and self.start[i] + self.duration[i] > tstop:
                istop = i
                set_stop = True
            if set_start and set_stop:
                break
        if not set_stop:
            istop = len(self.start)
        if not set_start:
            istart = istop
        self.start = self.start[istart:istop]
        self.duration = self.duration[istart:istop]


class Trace:
    def __init__(self, name, step_start, step_stop):
        self.step_start = step_start
        self.step_stop = step_stop
        self.rank = int(re.search("trace_(.*).txt", name).group(1))
        with open(name) as f:
            data = f.readlines()
        self.regions = {}
        current_region = ""
        for line in data:
            l = line.rstrip()
            words = l.split()
            if words[0] == "Region:":
                if words[1] == "StepTimer":
                    reg_name = words[1]
                    self.regions[reg_name] = Region()
                    current_region = reg_name
                    continue
                else:
                    fstr = l[l.find("::") + 2 :]
                    reg_name = fstr[fstr.find(":") + 2 :]
                    self.regions[reg_name] = Region()
                    current_region = reg_name
                    continue
            self.regions[current_region].add_samples(line)
        step_start_time = 0.0
        step_stop_time = 999999.0
        if step_start > 0:
            if step_start < len(self.regions["StepTimer"].start):
                step_start_time = self.regions["StepTimer"].start[step_start]
        if step_stop > -1 and step_stop < len(self.regions["StepTimer"].start):
            step_stop_time = (
                self.regions["StepTimer"].start[step_stop]
                + self.regions["StepTimer"].duration[step_stop]
            )
        for key, val in self.regions.items():
            if key == "StepTimer":
                continue
            val.trim(step_start_time, step_stop_time)

    def region_names(self):
        return list(self.regions.keys())

    def plot_trace(self, ax, colorMap, hatchMap):
        for key, val in self.regions.items():
            if key == "StepTimer":
                continue
            ax.barh(
                self.rank,
                val.duration,
                left=val.start,
                height=0.5,
                label=key,
                color=colorMap[key],
                hatch=hatchMap[key],
            )


def main(files, step_start, step_stop):
    trace = []
    for f in files:
        print("Getting trace", f, end="")
        trace.append(Trace(f, step_start, step_stop))
        print("  done!")
    # get max number of functions
    all_funcs = OrderedDict()
    for t in trace:
        for key in t.region_names():
            all_funcs[key] = ""

    num_colors = len(all_funcs)
    cm = plt.get_cmap("tab20")
    hatch = ["", "--", "/", "\\", "+", "x"]
    num_hatches = len(hatch)
    colorMap = {}
    hatchMap = {}
    cindex = 0
    for f, dum in all_funcs.items():
        colorMap[f] = cm((cindex + 0.5) / num_colors)
        hatchMap[f] = hatch[cindex % num_hatches]
        cindex += 1
    fig, ax = plt.subplots(figsize=(18, 12))
    for t in trace:
        print("Plotting trace", t.rank, end="")
        t.plot_trace(ax, colorMap, hatchMap)
        print("  done!")
    plt.xlabel("Time (s)")
    plt.ylabel("Rank")
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height*0.2, box.width, box.height * 0.8])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0, -0.02, 1, -0.02),
        ncol=3,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.files, args.step_start, args.step_stop)
