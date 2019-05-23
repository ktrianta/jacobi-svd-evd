import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from fractions import Fraction



beta = 16
pi = 16
I_intersect = pi / beta

evd_base = np.loadtxt("../evd/evd-base")
evd_eigen = np.loadtxt("../../results1/evd/eigen_evd_perf")
evd_oneloop = np.loadtxt("../evd/evd-oneloop")
evd_oneloop_vectorized = np.loadtxt("../evd/evd-oneloop-vectorized")
evd_outer_unroll = np.loadtxt("../evd/evd-outer-unroll")
evd_outer_unroll_vectorized = np.loadtxt("../evd/evd-outer-unroll-vectorized")
evd_base_vectorized= np.loadtxt("../evd/evd-vectorized")
evd_blocked= np.loadtxt("../../results1/evd/evd-blocked")
evd_blocked_vectorized= np.loadtxt("../evd/evd-blocked-vectorized")
evd_blocked_less_copy= np.loadtxt("../../results1/evd/evd-blocked-less-copy")
evd_blocked_less_copy_vectorized= np.loadtxt("../evd/evd-blocked-less-copy-vectorized")


svd_base = np.loadtxt("../svd/svd-base")
svd_blocked = np.loadtxt("../svd/svd-blocked")
svd_blocked_less_copy = np.loadtxt("../svd/svd-blocked-less-copy")
svd_blocked_less_copy_transposed = np.loadtxt("../svd/svd-blocked-less-copy-transposed")


def save(name):
    fig.savefig('{}.eps'.format(name), format='eps', dpi=1200, bbox_inches='tight')


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
yticks = [1/16,1/8,1/4,1/2,1, 2, 4, 8, 16, 32]
mem_x = [1/16, I_intersect]
mem_y = [1/16, pi]
cpu_x = [I_intersect, xticks[-1]]
cpu_y = [pi, pi]

ax.plot(mem_x, mem_y, linewidth=3, label=r'Bound based on $\beta={}$'.format(beta), color='b', alpha=0.4)
ax.plot(mem_x, cpu_y, ':', c='black')
ax.plot(cpu_x, cpu_y, linewidth=3, label=r'Bound based on $\pi={}$'.format(pi), color='r', alpha=0.4)
ax.plot([I_intersect, 2*I_intersect], [pi, beta*4*I_intersect], ':', c='black')
ax.plot([I_intersect, I_intersect], [0, pi], ':', c='black')

def add_cross(x, y, label,label2 = '', text_dx=0, text_dy=0):
    ax.plot(x, y, 'x-')
    ax.text(x[1] + text_dx, y[1] + text_dy, label2)
    ax.text(x[-1] + text_dx, y[-1] + text_dy, label)
    #ax.plot([0, x], [y, y], ':', c='black')
    #ax.plot([x, x], [yticks[0], y], ':', c='black')

#add_cross(evd_base[:,1],evd_base[:,0], 'base-evd 16', '1024')
#
#add_cross(evd_eigen[3:,1],evd_eigen[3:,0],'evd_eigen 16','1024')
#add_cross(evd_oneloop[3:,1],evd_oneloop[3:,0],'evd_oneloop 16','1024')
#add_cross(evd_oneloop_vectorized[3:,1],evd_oneloop_vectorized[3:,0],'evd_oneloop_vectorized 16','1024')
#add_cross(evd_outer_unroll[3:,1],evd_outer_unroll[3:,0],'evd_outer_unroll 16','1024')
#add_cross(evd_outer_unroll_vectorized[3:,0],evd_outer_unroll_vectorized[3:,0],'evd_outer_unroll_vectorized 16','1024')
#add_cross(evd_base_vectorized[3:,1],evd_base_vectorized[3:,0],'evd_base_vectorized 16','1024')

#add_cross(evd_blocked[:,0], evd_blocked[:,1],'evd_blocked 16', '1024')
#add_cross(evd_blocked_vectorized[:,0], evd_blocked_vectorized[:,1],'evd_blocked_vectorized 16', '1024')
#add_cross(evd_blocked_less_copy[:,0], evd_blocked_less_copy[:,1],'evd_blocked_less_copy 16', '1024')
#add_cross(evd_blocked_less_copy_vectorized[:,0], evd_blocked_less_copy_vectorized[:,1],'evd_blocked_less_copy_vectorized 16', '1024')

add_cross(svd_base[:,1],svd_base [:,0],'','svd_base 512')
add_cross(svd_blocked[:,1],svd_blocked [:,0],'','svd_base 512')
add_cross(svd_blocked_less_copy[:,1],svd_blocked_less_copy[:,0],'svd_blocked_less_copy16','svd_blocked_less_copy 512')
add_cross(svd_blocked_less_copy_transposed[:,1],svd_blocked_less_copy_transposed[:,0],'svd_blocked_less_copy_transposed16','512')

#add_cross(2, 6, 'Computation B')
#add_cross(xticks[-1] - 2, 10, 'Computation C', text_dx=-8)

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
save('svd-all-blocked')
