import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from fractions import Fraction


beta = 12
pi = 16
I_intersect = pi / beta


def save(name):
    fig.savefig('{}.pdf'.format(name), format='pdf', dpi=1200, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('I: Operational intensity [ops/bytes]')
ax.set_ylabel('P: Performance [ops/cycle]', rotation=0, ha='left')
ax.set_title('Roofline Plot', loc='left', fontdict={'weight':'bold'}, pad=25)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_label_coords(0, 1.02)
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=2)
ax.set_facecolor((0.9, 0.9, 0.9))

xticks = [1/16, 1/8, 1/4, I_intersect, 1/2, 1, 2, 4, 8, 16]
yticks = [1, 2, 4, 8, 16, 32]
mem_x = [0, I_intersect]
mem_y = [0, pi]
cpu_x = [I_intersect, xticks[-1]]
cpu_y = [pi, pi]

ax.plot(mem_x, mem_y, linewidth=3, label=r'Bound based on $\beta={}$'.format(beta), color='b', alpha=0.4)
ax.plot(mem_x, cpu_y, ':', c='black')
ax.plot(cpu_x, cpu_y, linewidth=3, label=r'Bound based on $\pi={}$'.format(pi), color='r', alpha=0.4)
ax.plot([I_intersect, 2*I_intersect], [pi, beta*2*I_intersect], ':', c='black')
ax.plot([I_intersect, I_intersect], [0, pi], ':', c='black')

def add_cross(x, y, label, text_dx=0, text_dy=-0.85):
    ax.plot(x, y, 'x', c='black')
    ax.text(x + text_dx, y + text_dy, label)
    #ax.plot([0, x], [y, y], ':', c='black')
    #ax.plot([x, x], [yticks[0], y], ':', c='black')

add_cross(1/2, 6, 'Computation A')
add_cross(2, 6, 'Computation B')
add_cross(xticks[-1] - 2, 10, 'Computation C', text_dx=-8)

ax.set_xlim([xticks[0], xticks[-1]])
ax.set_ylim([yticks[0], yticks[-1]])
plt.xticks(xticks, [Fraction(x).limit_denominator() for x in xticks])
plt.yticks(yticks, [Fraction(y).limit_denominator() for y in yticks])

def color_compute_bound_part():
    x_lo = I_intersect
    y_lo = yticks[0]
    width = xticks[-1] - x_lo
    height = pi - y_lo
    rect = patches.Rectangle((x_lo, y_lo), width, height, facecolor='r', alpha=0.2)
    ax.add_patch(rect)

def color_mem_bound_part():
    x_left = 1/beta
    x_right = I_intersect
    y_lo = yticks[0]
    y_hi = pi
    tri = patches.Polygon([(x_left, y_lo), (x_right, y_lo), (x_right, y_hi)], facecolor='b', alpha=0.2)
    ax.add_patch(tri)

color_compute_bound_part()
color_mem_bound_part()

ax.legend()
plt.show()
#save('roofline_simd_dots')
