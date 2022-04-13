import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import random
import itertools
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
import json

from const import CATEGORICAL, BROWS, STYPE, PLOTS_FP, SAVE_TO, RENAME_DICT
from transpose_openface import _get_mean_pos_frame

parser = argparse.ArgumentParser()
parser.add_argument("--openface_fp", default=os.path.join(SAVE_TO, 'open_face_transposed.csv'), type=str,
                    help="Path to a .csv file with transposed OpenFace")
parser.add_argument("--elan_stats_fp", default=os.path.join(SAVE_TO, 'additional_stats_from_elan.json'), type=str,
                    help="Path to a json file with computed means of the signs")
parser.add_argument("--targets", default=[
    'pose_Rx',
    'perp_dist39_42_3d_pred_MLP_best_model_diff',
    'perp_dist39_42_3d_pred_MLP_best_model_no_features_diff',
], type=list, nargs='+',
                    help="Which targets to plot")
parser.add_argument("--separate_brows_plot", default=False, type=bool,
                    help="Whether to plot brows on separate axis or in one")
parser.add_argument("--deaf", default=False, type=bool,
                    help="Whether to plot the mean deaf and hearing curves")
parser.add_argument("--save_to", default=os.path.join(SAVE_TO, PLOTS_FP), type=str,
                    help="Save path for logs of cross-validation")


def plot_mean(openface_fp, elan_stats_fp, targets, save_to, separate_brows_plot=False, deaf=False,
              rename_dict=RENAME_DICT):
    df = pd.read_csv(openface_fp, index_col=[0, 1])
    elan_stats = json.load(open(elan_stats_fp))
    start_pos_dict = _get_mean_pos_frame(elan_stats['pos_start'])
    end_pos_dict = _get_mean_pos_frame(elan_stats['pos_end'])

    if rename_dict is None:
        rename_dict = {}
    cmap = plt.get_cmap('Dark2')
    linestyles = ['-', '--', '-.']
    linewidth = 2.5

    x_axes=1
    if deaf:
        x_axes = 2
    if separate_brows_plot:
        x_axes = 2
    fig, axes = plt.subplots(len(targets), x_axes, figsize=(8 * x_axes, len(targets) * 4))

    if deaf:
        axes[0][0].set_title('deaf', fontsize='x-large')
        axes[0][1].set_title('hearing', fontsize='x-large')
    elif x_axes > 1:
        for i, brow in enumerate(BROWS):
            axes[0][i].set_title(brow.value, fontsize='x-large')

    for n, stype in enumerate(STYPE):
        color = cmap(n)
        style = linestyles[n]

        mask = df[CATEGORICAL.STYPE] == stype.value
        for i, metric in enumerate(targets):
            if 'pose' not in metric:
                if x_axes > 1:
                    if deaf:
                        mask_deaf = mask & df['deaf']
                        mask_hear = mask & ~df['deaf']
                        for j, new_mask in enumerate([mask_deaf, mask_hear]):
                            for k, brow in enumerate(BROWS):
                                index = (brow.value, metric, new_mask)
                                subset = df.loc[index, '0.0':'70.0'].mean()
                                subset.plot(ax=axes[i][j],
                                            color=color,
                                            linestyle=style,
                                            label=metric,
                                            linewidth=linewidth - k
                                            )
                    else:
                        for j, brow in enumerate(BROWS):
                            index = (brow.value, metric, mask)
                            subset = df.loc[index, '0.0':'70.0'].mean()
                            subset.plot(ax=axes[i][j],
                                        color=color,
                                        linestyle=style,
                                        label=metric,
                                        linewidth=linewidth)
                else:
                    df.loc[(BROWS.INNER.value,
                            metric,
                            mask), '0.0':'70.0'].mean().plot(ax=axes[i],
                                                             color=color,
                                                             label=BROWS.INNER.value,
                                                             linestyle=style,
                                                             linewidth=linewidth)
                    df.loc[(BROWS.OUTER.value,
                            metric,
                            mask), '0.0':'70.0'].mean().plot(ax=axes[i],
                                                             color=color,
                                                             label=BROWS.OUTER.value,
                                                             linestyle=style,
                                                             linewidth=linewidth - 1)
            else:
                if x_axes > 1:
                    if deaf:
                        mask_deaf = mask & df['deaf']
                        mask_hear = mask & ~df['deaf']
                        for j, new_mask in enumerate([mask_deaf, mask_hear]):
                            index = (BROWS.INNER.value, metric, new_mask)
                            df.loc[index, '0.0':'70.0'].mean().plot(ax=axes[i][j],
                                                                         color=color,
                                                                         linestyle=style,
                                                                         label=metric,
                                                                         linewidth=linewidth
                                                                         )
                            axes[i][j].invert_yaxis()
                    else:
                        for j in range(x_axes):
                            index = (BROWS.INNER.value, metric, mask)
                            df.loc[index, '0.0':'70.0'].mean().plot(ax=axes[i][j],
                                                                     color=color,
                                                                     linestyle=style,
                                                                     label=metric,
                                                                     linewidth=linewidth
                                                                     )
                            axes[i][j].invert_yaxis()
                else:
                    index = (BROWS.INNER.value, metric, mask)
                    df.loc[index, '0.0':'70.0'].mean().plot(ax=axes[i],
                                                             color=color,
                                                             linestyle=style,
                                                             label=metric,
                                                             linewidth=linewidth
                                                             )
                    axes[i].invert_yaxis()

            if x_axes > 1:
                axes[i][0].set_ylabel(rename_dict.get(metric, metric))
            else:
                axes[i].set_ylabel(rename_dict.get(metric, metric))

    for pos, start, end in zip(start_pos_dict.keys(),
                               start_pos_dict.values(),
                               end_pos_dict.values()):
        if pos == 'Q':
            continue
        for ax in axes.flatten():
            ax.axvline(start, alpha=0.4, color='red')
            ax.axvline(end, alpha=0.4, color='red')
            ax.text(np.mean([start, end]), ax.get_ylim()[0], pos,
                    horizontalalignment='center', verticalalignment='bottom',
                    fontsize='x-large', alpha=0.4, color='red')

    fig.legend(plt.gca().lines[::2], ['st', 'polar_q', 'wh_q'], loc='center', bbox_to_anchor=(0.35, 0.05), ncol=3,
               fontsize='medium')
    if deaf or x_axes == 1:
        fig.legend(plt.gca().lines[:2], ['inner', 'outer'], loc='center', ncol=2, bbox_to_anchor=(0.65, 0.05),
                   fontsize='medium')

    os.makedirs(save_to, exist_ok=True)
    ts = "-".join(targets)
    d = "-deaf" if deaf else ""
    b = "-brows" if separate_brows_plot else ""
    plt.savefig(os.path.join(save_to, f'mean_stypes-{ts}{d}{b}.png'))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    plot_mean(args.openface_fp, args.elan_stats_fp, args.targets, args.save_to, args.separate_brows_plot, args.deaf)
