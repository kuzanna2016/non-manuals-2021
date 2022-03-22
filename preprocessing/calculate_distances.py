import os
import pandas as pd
import json
import argparse

from const import SAVE_TO, CONFIGS
from distances import find_mean_dist, find_mean_perp_dist, find_mean_perp_plane_dist

parser = argparse.ArgumentParser()
parser.add_argument("--openface_fp", default=os.path.join(SAVE_TO, 'open_face_data_combined.csv'), type=str,
                    help="Path to a .csv file with combined OpenFace results")
parser.add_argument("--config_fp", default=os.path.join(CONFIGS, 'distances.json'), type=str,
                    help="Path to a json file with distances configuration")
parser.add_argument("--override", default=False, type=bool,
                    help="To override previously calculated distances")
parser.add_argument("--save_to", default=SAVE_TO, type=str, help="Save folder path")


def compute_distances(df, distances_config, override=False, plot_face=False):
    for dist, params in distances_config.items():
        for points in params:
            inner = points['inner']
            outer = points['outer']
            if 'perp' in dist:
                if 'line' in dist:
                    df = find_mean_perp_dist(df,
                                             inner_brows=inner,
                                             outer_brows=outer,
                                             point_1=points['perp'][0],
                                             point_2=points['perp'][1],
                                             override=override)
                else:
                    df = find_mean_perp_plane_dist(df,
                                                   inner_brows=inner,
                                                   outer_brows=outer,
                                                   perp_points=points['perp'],
                                                   override=override,
                                                   plot_face=plot_face)
            else:
                df = find_mean_dist(df,
                                    inner_brows=inner,
                                    outer_brows=outer,
                                    point_2=points['point'][0],
                                    override=override)
    return df


def calculate_distances(openface_fp, config_fp, override, save_to):
    df = pd.read_csv(openface_fp, index_col=[0, 1])
    distances_config = json.load(open(config_fp))
    df = compute_distances(df, distances_config, override=override)
    df.to_csv(os.path.join(save_to, 'open_face_with_distances.csv'))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    calculate_distances(openface_fp=args.openface_fp, config_fp=args.config_fp, override=args.override, save_to=args.save_to)
