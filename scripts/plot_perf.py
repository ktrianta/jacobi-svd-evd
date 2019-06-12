import sys
import ntpath
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def save(name):
    fig.savefig('{}.eps'.format(name), format='eps', dpi=1200, bbox_inches='tight')


input_filename = sys.argv[1]
perf_filenames = sys.argv[2:]

cpu_peak_perf = 16
cpu_peak_perf_no_vec = 4

input_sizes = np.loadtxt(input_filename)
test_names = perf_filenames
test_perfs = [np.loadtxt(f) for f in perf_filenames]

# There may exist results for all input sizes, so ignore the input sizes for which no results exist.
results_max_count = max([len(p) for p in test_perfs])
input_sizes = input_sizes[ : min(len(input_sizes), results_max_count)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Matrix size')
ax.set_ylabel('[Flops/cycle]', rotation=0, ha='left')
ax.set_title('Performance                        Core i7-7600U @ 2.8GHz, GCC 7.4.0 -O3 -mavx2 -mfma', loc='left', fontdict={'weight':'bold'}, pad=25)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_label_coords(0, 1.02)

ax.set_facecolor((0.9, 0.9, 0.9))
plt.grid(color='w', axis='y')
plt.ylim([0, cpu_peak_perf + 1])
plt.xticks(input_sizes)

# log-scale x 
ax.set_xscale('log', basex=2)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.margins(x=0.04)

cpu_x = [input_sizes[0], input_sizes[-1]]
cpu_y = [cpu_peak_perf, cpu_peak_perf]
ax.plot(cpu_x, cpu_y, linewidth=3, label=r'peak performance bound', color='r', alpha=0.4)

for name, perf in zip(test_names, test_perfs):
    basename = ntpath.basename(name)  # remove the path and keep only the basename of the file
    basename = basename.replace("-", " ")  # replace dashes with spaces
    if basename == "base" or basename == "svd-base":
        line, = ax.plot(input_sizes[:len(perf)], perf, '-o', label=basename, color="#1f77b4")
    elif basename == "eigen" or basename == "svd-eigen":
        line, = ax.plot(input_sizes[:len(perf)], perf, '-o', label=basename, color="#e377c2")
    elif basename == "base-vectorized":
        line, = ax.plot(input_sizes[:len(perf)], perf, '-o', label=basename, color="#ff7f0e")
    elif basename == "blocked-vectorized":
        line, = ax.plot(input_sizes[:len(perf)], perf, '-o', label=basename, color="#2ca02c")
    elif basename == "blocked-vectorized-less-data":
        line, = ax.plot(input_sizes[:len(perf)], perf, '-o', label=basename, color="#d62728")
    elif basename == "blocked-vectorized-less-data-opt-mult":
        line, = ax.plot(input_sizes[:len(perf)], perf, '-o', label=basename, color="#8c564b")
    else:
        line, = ax.plot(input_sizes[:len(perf)], perf, '-o', label=basename)
    line.set_clip_on(False)

plt.legend(loc="best", bbox_to_anchor=(1.0, 0.7))
plt.show()
