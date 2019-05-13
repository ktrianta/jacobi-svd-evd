import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def save(name):
    fig.savefig('{}.eps'.format(name), format='eps', dpi=1200, bbox_inches='tight')


input_filename = sys.argv[1]
perf_filenames = sys.argv[2:]

cpu_peak_perf_vec = 16
cpu_peak_perf_no_vec = 4

input_sizes = np.loadtxt(input_filename)
test_names = ['CPU peak vec'] + perf_filenames
test_perfs = [cpu_peak_perf_vec*np.ones_like(input_sizes)] + [np.loadtxt(f) for f in perf_filenames]

# Enable also peak performance without vector instructions
# test_names = ['CPU peak vec', 'CPU peak no-vec'] + perf_filenames
# test_perfs = [cpu_peak_perf_vec*np.ones_like(input_sizes), cpu_peak_perf_no_vec*np.ones_like(input_sizes)] +
#     [np.loadtxt(f) for f in perf_filenames]

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Input size')
ax.set_ylabel('[Flops/cycle]', rotation=0, ha='left')
ax.set_title('Performance', loc='left', fontdict={'weight':'bold'}, pad=25)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_label_coords(0, 1.02)

ax.set_facecolor((0.9, 0.9, 0.9))
ax.set_yticks(np.arange(0, np.max(test_perfs) + 1, 1))
plt.grid(color='w', axis='y')
plt.ylim([0, cpu_peak_perf_vec + 1])
plt.xticks(input_sizes)

# log-scale x 
ax.set_xscale('log', basex=2)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.margins(x=0.04)

for name, perf in zip(test_names, test_perfs):
    line, = ax.plot(input_sizes, perf, '-o', label=name)
    line.set_clip_on(False)

plt.legend()
plt.show()
