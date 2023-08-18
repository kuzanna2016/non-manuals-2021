import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import numpy as np

import random

from const import BROWS, PLOTS_FP, SAVE_TO

parser = argparse.ArgumentParser()
parser.add_argument("--openface_fp", default=os.path.join(SAVE_TO, 'open_face_with_bias_correction.csv'), type=str,
                    help="Path to a .csv file with OpenFace")
parser.add_argument("--elan_fp", default=os.path.join(SAVE_TO, 'elan_preprocessed.tsv'), type=str,
                    help="Path to an Elan file with annotations")
parser.add_argument("--sentences", type=str, nargs='*',
                    help="Sentences to plot")
parser.add_argument("--targets", default=[
    'perp_dist39_42_3d_pred_MLP_best_model_diff',
    'perp_dist39_42_3d_pred_MLP_best_model_no_features_diff',
], type=list, nargs='+',
                    help="Which targets to plot")
parser.add_argument("--n_samples", default=5, type=int,
                    help="If no video name, random n_samples will be plotted")
parser.add_argument("--brows", default=BROWS.INNER.value, type=str, nargs='+',
                    help="The eyebrow distance to plot [inner, outer]")
parser.add_argument("--plot_head", default='', type=str,
                    help="Whether to plot the head rotation, specify which axis if plot, like 'x', or 'xy' or 'xyz'")
parser.add_argument("--save_to", default=os.path.join(SAVE_TO, PLOTS_FP), type=str,
                    help="Save path for logs of cross-validation")


def plot_samples(openface_fp,
                 elan_fp,
                 targets,
                 save_to,
                 sentences=None,
                 n_samples=5,
                 brows=[BROWS.INNER.value],
                 plot_head=False):
    df = pd.read_csv(openface_fp, index_col=[0, 1])
    elan = pd.read_csv(elan_fp, sep='\t')
    if sentences is None:
        samples = random.sample(df.index.levels[0].tolist(), n_samples)
    else:
        samples = sentences

    n_cols = len(targets) + len(plot_head) if plot_head else len(targets)
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)
    for n, sample in enumerate(samples):
        j = -1
        for j, target in enumerate(targets):
            for brow in brows:
                name = f'{brow}_{target}'
                axes[n][j].plot(df.loc[sample, name])
            axes[0][j].set_title(target)
        if plot_head:
            for k, pose_ax in enumerate(plot_head):
                name = f'pose_R{pose_ax}'
                axes[n][j + 1 + k].plot(df.loc[sample, name])
                axes[n][j + 1 + k].set_ylim(-1,1)
                if pose_ax == 'x':
                    axes[n][j + 1 + k].invert_yaxis()
                axes[0][j + 1 + k].set_title(name)
        axes[n][0].set_ylabel(sample)

        mask = elan.video_name == sample
        if elan[mask].brows.notna().any():
            br_moves = elan[mask].dropna(subset=['brows'])[
                ['start_frames', 'end_frames']].values
            for k, (start, end) in enumerate(br_moves):
                for i in range(n_cols):
                    axes[n][i].axvline(start, alpha=0.5, color='red')
                    axes[n][i].axvline(end, alpha=0.5, color='red')
                    axes[n][i].text(x=start + (end - start) / 2,
                                    y=axes[n][i].get_ylim()[0],
                                    s=elan[mask]['brows'].dropna().iloc[k],
                                    horizontalalignment='center', verticalalignment='bottom',
                                    color='red',
                                    fontsize=10,
                                    alpha=0.5,
                                    )
    for ax in axes.flatten():
        ax.set_xlabel('')
        ax.tick_params(reset=True)

    brows_str = '_'.join(brows)
    if sentences:
        sentences_str = '_'.join(sentences)
        f_name = os.path.join(save_to, f'{sentences_str}_{brows_str}_compare_targets')
    else:
        f_name = os.path.join(save_to, f'{n_samples}_{brows_str}_samples_compare_changes')
    i = 1
    while os.path.isfile(f_name + '_' + str(i) + '.png'):
        i += 1
    plt.savefig(f_name + '_' + str(i) + '.png', bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    plot_samples(args.openface_fp, args.elan_fp, targets=args.targets, sentences=args.sentences, n_samples=args.n_samples, brows=args.brows,
                 plot_head=args.plot_head, save_to=args.save_to)
