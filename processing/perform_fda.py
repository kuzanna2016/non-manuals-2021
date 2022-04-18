import os
import json
import pandas as pd
import argparse
import datetime
from collections import defaultdict
import numpy as np

from skfda.preprocessing.dim_reduction.projection import FPCA
from skfda.representation.basis import BSpline
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.registration import landmark_elastic_registration
from skfda import FDataGrid

from plot_fpca import plot_significant_components, plot_registered_curves, plot_perturbation_graph
from const import SAVE_TO, CONFIGS_FP, SPEAKERS, FDA

parser = argparse.ArgumentParser()
parser.add_argument("--openface_fp", default=os.path.join(SAVE_TO, 'open_face_with_bias_correction.csv'), type=str,
                    help="Path to a .csv file with OpenFace outputs with computed distances")
parser.add_argument("--pos_boundaries_fp", default=os.path.join(SAVE_TO, 'pos_boundaries.csv'), type=str,
                    help="Path to a csv file with part of speech frame boundaries")
parser.add_argument("--configs_fp", default=os.path.join(CONFIGS_FP, 'fda.json'), type=str,
                    help="Path to a .json file with the fda smoothing parameters")
parser.add_argument("--save_to", default=os.path.join(SAVE_TO, FDA), type=str,
                    help="Save path for the computed pc scores")


def perform_fda(openface_fp, pos_boundaries_fp, configs_fp, save_to):
    df = pd.read_csv(openface_fp, index_col=[0, 1])

    boundaries = pd.read_csv(pos_boundaries_fp, index_col=0, header=[0, 1])
    video_names = boundaries.index.unique()

    configs = json.load(open(configs_fp))
    for config in configs:
        experiment_name = config.get("experiment_name", datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        print('Running experiment', experiment_name)
        names = config.get("names")
        n_basis = config.get("n_basis")
        order = config.get("order")
        n_fpca = config.get("n_fpca")
        rename_dict = config.get("rename_dict", {})
        fpcs = config.get("fpcs_plots", {})
        path = os.path.join(save_to, experiment_name)
        os.makedirs(path, exist_ok=True)

        variables, sentence_type, speaker_ids, sentences, norm_rng, durations, t_norms = create_variables(df, names,
                                                                                                          video_names)
        fds_variables = {name: create_fds(data, t_norms) for name, data in variables.items()}
        basis = BSpline(domain_range=norm_rng, n_basis=n_basis, order=order)
        fd_basis_variables = {
            name: to_basis_fds(data, basis, w_basis_smoother=True)
            for name, data in fds_variables.items()
        }

        boundaries = boundaries.loc[
            video_names, [('start_frames', 'N'), ('end_frames', 'N'), ('start_frames', 'V'), ('end_frames', 'V')]]
        norm_boundaries = (boundaries.values.T / np.array(durations)) * np.mean(durations)
        norm_boundaries = norm_boundaries.T
        landmark_location = norm_boundaries.mean(axis=0)
        fd_registered_variables = {name: align_landmarks(data, norm_boundaries, landmark_location) for name, data in
                                   fd_basis_variables.items()}
        plot_registered_curves(fd_basis_variables, fd_registered_variables, names, sentence_type, rename_dict,
                               landmark_location, path, experiment_name)

        fpca_variables = {name: get_fpca(data, n_fpca) for name, data in fd_registered_variables.items()}
        variance_results = {}
        print('Explained variance ratio')
        for name in names:
            print(f'for {name}:')
            fpca = fpca_variables[name]
            variance_results[name] = {
                j: component_variance
                for j, component_variance in enumerate(fpca.explained_variance_ratio_)
            }
            variance = ['PC{} {:.0%}'.format(j + 1, component_variance) for j, component_variance in
             enumerate(fpca.explained_variance_ratio_)]
            print('\t'.join(variance))
        json.dump(variance_results, open(os.path.join(path, 'explained_variance_ratio.json'), 'w'))

        scores_variables = {name: get_scores(data, fpca_variables[name]) for name, data in
                            fd_registered_variables.items()}
        plot_perturbation_graph(names, n_fpca, fpca_variables, scores_variables, landmark_location, rename_dict,
                                path,
                                experiment_name)

        deaf = ['deaf' if speaker in SPEAKERS.DEAF else 'hearing' for speaker in speaker_ids]
        for name, score in scores_variables.items():
            scores = pd.DataFrame(score, columns=[f'PC_{i + 1}' for i in range(score.shape[1])],
                                  index=video_names)
            scores['sType'] = sentence_type
            scores['deaf'] = np.asarray(deaf)
            scores['speaker_id'] = speaker_ids
            scores['sentence'] = sentences
            scores.to_csv(os.path.join(path, f'{name}_fpca_scores.csv'))
            plot_configs = fpcs.get(name, [])
            for c in plot_configs:
                components = c.get("components")
                plot_deaf = c.get("plot_deaf", False)
                plot_significant_components(fpca_variables, scores, components, landmark_location, name, plot_deaf, path,
                                            rename_dict)



def create_variables(df, names, video_names):
    variables = defaultdict(list)
    for name in names:
        for video_name in video_names:
            data = df.loc[video_name, name]
            variables[name].append(data.values)

    frameids = []
    for video_name in video_names:
        data = df.loc[video_name].index
        frameids.append(data.values)

    sentence_type = []
    speaker_ids = []
    sentences = []
    durations = []
    for video_name in video_names:
        sentence_type.append(df.loc[video_name, 'sType'].iloc[0])
        speaker_ids.append(df.loc[video_name, 'speaker'].iloc[0])
        sentences.append(df.loc[video_name, 'sentence'].iloc[0])
        durations.append(df.loc[video_name].index.values.max())
    sentence_type = np.asarray(sentence_type)
    speaker_ids = np.asarray(speaker_ids)
    sentences = np.asarray(sentences)
    norm_rng = (0, np.mean(durations))
    durations = np.asarray(durations)

    t_norms = (frameids / durations) * np.mean(durations)
    return variables, sentence_type, speaker_ids, sentences, norm_rng, durations, t_norms


def create_fds(data, t_norms):
    fds = [
        FDataGrid(
            data_matrix=d,
            grid_points=t_norms[i],
        )
        for i, d in enumerate(data)
    ]
    return fds


def smooth_fds(fds, smoother):
    fds_smooth = []
    for fd in fds:
        fd_smooth = smoother.fit_transform(fd)
        fds_smooth.append(fd_smooth)
    return fds_smooth


def to_basis_fds(fds, basis, w_basis_smoother=False):
    if w_basis_smoother:
        smoother = BasisSmoother(basis, smoothing_parameter=1, return_basis=True)
        fds_basis = [
            smoother.fit_transform(fd)
            for fd in fds
        ]
    else:
        fds_basis = [
            fd.to_basis(basis)
            for fd in fds
        ]
    all_fd_basis = fds_basis[0]
    for fd_basis in fds_basis[1:]:
        all_fd_basis = all_fd_basis.concatenate(fd_basis)
    all_fd_basis.extrapolation = 'bounds'
    return all_fd_basis


def align_landmarks(fd_basis, boundaries, landmark_location):
    fd_registered = landmark_elastic_registration(fd_basis.to_grid(),
                                                  landmarks=boundaries,
                                                  location=landmark_location)
    return fd_registered


def get_fpca(fd, n=5):
    fpca = FPCA(n_components=n)
    fpca = fpca.fit(fd)
    return fpca


def get_scores(fd, fpca):
    scores = fpca.transform(fd)
    return scores


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    perform_fda(args.openface_fp, args.pos_boundaries_fp, args.configs_fp, save_to=args.save_to)
