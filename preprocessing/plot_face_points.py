import matplotlib.pyplot as plt
import itertools
import argparse
import os
import pandas as pd
import numpy as np

import random
from const import PLOTS_FP, SAVE_TO

parser = argparse.ArgumentParser()
parser.add_argument("--openface_fp", default=os.path.join(SAVE_TO, 'open_face_with_bias_correction.csv'), type=str,
                    help="Path to a .csv file with OpenFace")
parser.add_argument("--sentence", default=None, type=str,
                    help="Sentence to plot, if None, and pick_random_n is not None, picks pick_random_n frames from the whole datacet")
parser.add_argument("--frame", default=None, type=int,
                    help="Frame to plot, if None and sentence is not None and pick_random_n is not None, picks pick_random_n from sentence")
parser.add_argument("--pick_random_n", default=None, type=int,
                    help="Number of frames to pick, if None and frame is None - all frames from sentence will be plotted")
parser.add_argument("--view_points", default=1, type=int,
                    help="Number of view points")
parser.add_argument("--do_all_around_view", default=False, type=bool,
                    help="If true will plot all view points")
parser.add_argument("--save_to", default=os.path.join(SAVE_TO, PLOTS_FP), type=str,
                    help="Save path for the plot")


def plot(frames, sentence, save_to, view_points=1, frame_step_size=20, elev=200, azim=40, order='zxy'):
    coords = {
        'x': frames.filter(regex='^X_\d?\d').iloc[:, np.r_[17:27, 36:48]],
        'y': frames.filter(regex='^Y_\d?\d').iloc[:, np.r_[17:27, 36:48]],
        'z': frames.filter(regex='^Z_\d?\d').iloc[:, np.r_[17:27, 36:48]],
    }
    values = [coords[c].values for c in order]
    n_frames = frames.shape[0]

    cmap = plt.get_cmap('hot')
    fig = plt.figure(figsize=(5 * view_points, 5))
    for i in range(view_points):
        ax = fig.add_subplot(1, view_points, i + 1, projection='3d')
        for n, frame in enumerate(zip(*values)):
            color = cmap(n / n_frames)
            frame = list(frame)
            frame[2] += n * frame_step_size
            ax.scatter(*frame, color=color)
        ax.set_xlabel(order[0])
        ax.set_ylabel(order[1])
        ax.set_zlabel(order[2])
        ax.view_init(elev=elev, azim=(i + 1) * azim)

    fig.suptitle(sentence)
    f_name = f'{sentence if sentence is not None else "random"}_eyebrow_movement'
    i = 1
    while os.path.isfile(os.path.join(save_to, f_name + '_' + str(i) + '.png')):
        i += 1
    plt.savefig(os.path.join(save_to, f_name + '_' + str(i) + '.png'), bbox_inches='tight')


def plot_360(frames, sentence, save_to, step=40, rotation=720):
    coords = {
        'x': frames.filter(regex='^X_\d?\d').iloc[:, np.r_[17:27, 36:48]],
        'y': frames.filter(regex='^Y_\d?\d').iloc[:, np.r_[17:27, 36:48]],
        'z': frames.filter(regex='^Z_\d?\d').iloc[:, np.r_[17:27, 36:48]],
    }
    n_frames = frames.shape[0]

    cmap = plt.get_cmap('hot')
    shape = (rotation // step) + 1

    for order in itertools.permutations('xyz', 3):
        print('Plotting', "".join(order))
        values = [coords[c].values for c in order]
        fig = plt.figure(figsize=(5 * shape, 5 * shape))
        for i in range(0, rotation + 1, step):
            for j in range(0, rotation + 1, step):
                ix = ((i // step) * shape) + (j // step) + 1
                ax = fig.add_subplot(shape, shape, ix, projection='3d')
                for n, frame in enumerate(zip(*values)):
                    color = cmap(n / n_frames)
                    frame = list(frame)
                    ax.scatter(*frame, color=color)
                ax.set_xlabel(order[0])
                ax.set_ylabel(order[1])
                ax.set_zlabel(order[2])
                ax.view_init(elev=i, azim=j)
                ax.set_title(f'e{i},a{j}')

        fig.suptitle(sentence)
        f_name = f'360_rotation_{"".join(order)}_{sentence}'
        i = 1
        while os.path.isfile(os.path.join(save_to, f_name + '_' + str(i) + '.png')):
            i += 1
        plt.savefig(os.path.join(save_to, f_name + '_' + str(i) + '.png'), bbox_inches='tight')


def plot_faces(openface_fp, save_to, sentence=None, frame=None, pick_random_n=None, view_points=1,
               do_all_around_view=False):
    df = pd.read_csv(openface_fp, index_col=[0, 1])
    if sentence is not None:
        video = df.loc[sentence]
        if frame is not None:
            video = video.loc[frame]
            video = video.to_frame().T
        elif pick_random_n is not None:
            ixs = random.choices(np.arange(video.shape[0]), k=pick_random_n)
            video = video.iloc[ixs]
    elif pick_random_n is not None:
        ixs = random.choices(np.arange(df.shape[0]), k=pick_random_n)
        video = df.iloc[ixs]
    else:
        sentence = random.choice(df.index.levels[0])
        video = df.loc[sentence]

    if video.ndim == 1:
        video = video.to_frame().T
    if do_all_around_view:
        plot_360(video, sentence, save_to)
    else:
        plot(video, sentence, save_to, view_points)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    plot_faces(args.openface_fp, args.save_to, args.sentence, args.frame, args.pick_random_n, args.view_points,
               args.do_all_around_view)
