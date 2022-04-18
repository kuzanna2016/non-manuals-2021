import os
import numpy as np
import matplotlib.pyplot as plt
from skfda.exploratory.visualization import FPCAPlot

FONTSIZE = 20
LINEWIDTH = 4
ALPHA = 0.4
cmap = plt.get_cmap('Dark2')


def plot_significant_components(fpca_variables, scores, components, landmark_location, name, plot_deaf, save_to, rename_dict):
    fig, axes = plt.subplots(1, len(components), figsize=(7 * (len(components)), 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i, component in enumerate(components):
        fpca = fpca_variables[name]
        fpca.mean_.plot(axes=axes[i], linestyle='--', label='mean', linewidth=LINEWIDTH, c=cmap(7))
        if plot_deaf:
            for j, deaf in enumerate(scores.deaf.unique()):
                mean_score = np.mean(scores.loc[scores.deaf == deaf].iloc[:, component])
                curve = fpca.mean_ + mean_score * fpca.components_[component]
                curve.plot(axes=axes[i], label=deaf, linewidth=LINEWIDTH, c=cmap(j))
        else:
            for j, stype in enumerate(scores.sType.unique()):
                mean_score = np.mean(scores.loc[scores.sType == stype].iloc[:, component])
                curve = fpca.mean_ + mean_score * fpca.components_[component:component + 1]
                curve.plot(axes=axes[i], label=rename_dict[stype], linewidth=LINEWIDTH, c=cmap(j))
        axes[i].set_title(f'PC{component + 1}', fontsize=FONTSIZE + 2)

    axes[0].set_ylabel(rename_dict.get(name, name), fontsize=FONTSIZE + 4)
    for ax in axes:
        if 'pose_Rx' in name:
            ax.invert_yaxis()
        y_lim = ax.get_ylim()
        height = np.abs(y_lim[0] - y_lim[1])
        ax.vlines(landmark_location, *y_lim, color=cmap(3), alpha=ALPHA, linewidth=LINEWIDTH)
        ax.text(np.mean(landmark_location[:2]), y_lim[1] + height * 0.05, 'N', fontsize=FONTSIZE, color=cmap(3),
                alpha=ALPHA, horizontalalignment='center', verticalalignment='bottom')
        ax.text(np.mean(landmark_location[2:]), y_lim[1] + height * 0.05, 'V', fontsize=FONTSIZE, color=cmap(3),
                alpha=ALPHA, horizontalalignment='center', verticalalignment='bottom')
    fig.subplots_adjust(bottom=0.2)
    fig.legend(axes[0].lines, [l.get_label() for l in axes[0].lines], ncol=4, bbox_to_anchor=(0.5, 0),
               loc='lower center', fontsize=FONTSIZE)
    c_str = '_'.join(f'PC{c + 1}' for c in components)
    deaf = "_deaf_hearing" if plot_deaf else ""
    plt.savefig(os.path.join(save_to, f'{name}_{c_str}{deaf}.png'), bbox_inches='tight')


def plot_perturbation_graph(names, n_fpca, fpca_variables, scores_variables, landmark_location, rename_dict, save_to,
                            experiment_name):
    fig, axes = plt.subplots(len(names[:-1]), n_fpca, figsize=(n_fpca * 7, len(names[:-1]) * 5))
    for i, name in enumerate(names[:-1]):
        fpca = fpca_variables[name]
        multiple = float(np.std(scores_variables[name]))
        FPCAPlot(
            fpca.mean_,
            fpca.components_,
            multiple,
            axes=axes[i, :].tolist()
        ).plot()
        for ax in axes[i, :]:
            ax.set_title('')
            y_lim = ax.get_ylim()
            height = np.abs(y_lim[0] - y_lim[1])
            ax.vlines(landmark_location, *y_lim, color=cmap(3), alpha=ALPHA, linewidth=LINEWIDTH)
            ax.text(np.mean(landmark_location[:2]), y_lim[0] + height * 0.05, 'N', fontsize=FONTSIZE, color=cmap(3),
                    alpha=ALPHA, horizontalalignment='center', verticalalignment='bottom')
            ax.text(np.mean(landmark_location[2:]), y_lim[0] + height * 0.05, 'V', fontsize=FONTSIZE, color=cmap(3),
                    alpha=ALPHA, horizontalalignment='center', verticalalignment='bottom')
        axes[i, 0].set_ylabel(rename_dict.get(name, name), fontsize=FONTSIZE + 4)

    for i in range(n_fpca):
        axes[0, i].set_title(f'PC{i + 1}', fontsize=FONTSIZE + 2)
        axes[0, i].invert_yaxis()

    for ax in axes.flatten():
        for ln in ax.get_lines():
            c = ln.get_color()
            if c == '#ff7f0e':
                ln.set_color(cmap(1))
                ln.set_marker('+')
                ln.set_linestyle(' ')
                ln.set_markevery(25)
                ln.set_markersize(15)
                ln.set_markeredgewidth(LINEWIDTH)
            elif c == '#2ca02c':
                ln.set_color(cmap(0))
                ln.set_marker('_')
                ln.set_linestyle(' ')
                ln.set_markevery(25)
                ln.set_markersize(15)
                ln.set_markeredgewidth(LINEWIDTH)
            else:
                ln.set_color(cmap(7))
                ln.set(linewidth=LINEWIDTH)
    plt.savefig(os.path.join(save_to, experiment_name + '_perturbation_graph.png'), bbox_inches='tight')


def plot_registered_curves(fd_basis_variables, fd_registered_variables, names, sentence_type, rename_dict,
                           landmark_location, save_to, experiment_name):
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 5))

    for i, name in enumerate(names):
        for j, stype in enumerate(np.unique(sentence_type)):
            mask = sentence_type == stype
            fd = fd_basis_variables[name][mask].mean()
            fd_shift = fd_registered_variables[name][mask].mean()
            fd.plot(axes=axes[i], color=cmap(j), linestyle='--', label=f'{rename_dict[stype]}', linewidth=LINEWIDTH)
            fd_shift.plot(axes=axes[i], color=cmap(j), label=f'{rename_dict[stype]}_registered', linewidth=LINEWIDTH)

        y_lim = axes[i].get_ylim()
        height = np.abs(y_lim[0] - y_lim[1])
        axes[i].vlines(landmark_location, *y_lim, color=cmap(3), alpha=ALPHA, linewidth=LINEWIDTH)
        axes[i].text(np.mean(landmark_location[:2]), y_lim[0] + height * 0.05, 'N', fontsize=FONTSIZE, color=cmap(3),
                     alpha=ALPHA, horizontalalignment='center', verticalalignment='bottom')
        axes[i].text(np.mean(landmark_location[2:]), y_lim[0] + height * 0.05, 'V', fontsize=FONTSIZE, color=cmap(3),
                     alpha=ALPHA, horizontalalignment='center', verticalalignment='bottom')
        axes[i].set_title(rename_dict.get(name, name), fontsize=FONTSIZE)
    axes[0].invert_yaxis()
    fig.legend(plt.gca().lines, ['polar_q', 'polar_q_registered', 'wh_q', 'wh_q_registered', 'st', 'st_registered'],
               ncol=3, bbox_to_anchor=(0.5, -0.07), loc='lower center', fontsize=FONTSIZE)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(save_to, experiment_name + '_registered_landmarks.png'), bbox_inches='tight')
