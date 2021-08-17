import math
import sys

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors


def heat_map(matrix, x_label, y_label, ax,
             title=None, center=None, vmax=None, color=None, log=False, mask_zeros=False):
    if len(matrix[0]) == 11:
        columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'all']
    else:
        columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # for values with no data, annotate with "*", then plot heatmap over it
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if np.isnan(matrix[i][j]):
                ax.text(j+0.22, i+0.75, "   no\ninputs", size=15)

    # mask cases of C == MC
    for c in range(10):
        matrix[c][c] = None
    if mask_zeros:
        mask = (matrix == 0)
        matrix[mask] = None
    color='copper_r'
    if log is False:
        matrix = pd.DataFrame(matrix, columns=columns)
        sns.heatmap(matrix, linewidths=.5, center=center, robust=True, ax=ax, vmax=vmax, cmap=color,
                    annot=True, fmt='', linecolor="black")
        ax.set(title=title, xlabel=x_label, ylabel=y_label)
    else:
        min_v = pow(10, -7) if np.nanmin(matrix) == 0 else np.nanmin(matrix)
        max_v = np.nanmax(matrix) if vmax is None else vmax
        log_norm = mcolors.PowerNorm(gamma=0.4, vmin=min_v, vmax=max_v)
        # cbar_ticks = [math.pow(10, i) for i in
        #               range(math.floor(math.log10(min_v)), 1 + math.ceil(math.log10(max_v)))]
        cbar_ticks = [0.01, 0.1, 0.5, 1, 3, 5, 7, 10]
        matrix = pd.DataFrame(matrix, columns=columns)
        sns.heatmap(matrix, linewidths=.5, center=center, robust=True, ax=ax, vmax=vmax, cmap=color,
                    norm=log_norm, cbar_kws={"ticks": cbar_ticks}, annot=True, fmt='', linecolor="black")
        ax.set(title=title, xlabel=x_label, ylabel=y_label)
    # cross out diagonal of C==MC
    ax.plot(range(11), range(11), color="black")
    for c in range(11):
        ax.plot([c, c+1], [c+1, c], color="black")


def plot_heatmaps(values, names, shape, suptitle, vmaxs=None, colors=None, log=None, mask_zeros=None):
    if vmaxs is None:
        vmaxs = [None]*len(values)
    if colors is None:
        colors = [None] * len(values)
    if log is None:
        log = [False] * len(values)
    if mask_zeros is None:
        mask_zeros = [False] * len(values)
    fig, axs = plt.subplots(shape[0], shape[1], figsize=(15, 10))
    if shape == (1, 1):
        flat_axs = [axs]
    else:
        flat_axs = axs.flatten()
    for i in range(len(values)):
        sns.set_theme(context='paper', style='ticks', font_scale=3)
        heat_map(values[i], names[i][0], names[i][1], ax=flat_axs[i], title=names[i][2], center=None,
                 vmax=vmaxs[i], color=colors[i], log=log[i], mask_zeros=mask_zeros[i])
    fig.suptitle(suptitle, fontsize=20)


def plot_3d_bars(values, names, shape, suptitle):
    fig = plt.figure(figsize=plt.figaspect(1.0/len(values)))
    for i in range(len(values)):
        ax = fig.add_subplot(1, len(values), i+1, projection='3d')
        vals = values[i].flatten()
        cmap = plt.cm.get_cmap('inferno')
        colors = [cmap(k / float(vals.max())) for k in vals]
        ax.bar3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]*10, [el for e in range(10) for el in [e+1]*10], np.zeros(100),
                 np.ones(100), np.ones(100), vals, color=colors)
        ax.set(title=names[i][2], xlabel=names[i][0], ylabel=names[i][1])
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20)


# save the images
def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


# save the csv values
def save_csv_vals(values, csv_names, complete_name=False):
    for i in range(len(values)):
        if not complete_name:
            np.savetxt(f"csv_values/{csv_names[i]}", values[i], delimiter=',')
        else:
            np.savetxt(f"{csv_names[i]}", values[i], delimiter=',')
