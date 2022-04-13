import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict

from const import CATEGORICAL, SAVE_TO, SPEAKERS, INDEX

parser = argparse.ArgumentParser()
parser.add_argument("--openface_fp", default=os.path.join(SAVE_TO, 'open_face_with_bias_correction.csv'), type=str,
                    help="Path to a .csv file with OpenFace outputs with computed distances")
parser.add_argument("--elan_fp", default=os.path.join(SAVE_TO, 'elan_preprocessed.tsv'), type=str,
                    help="Path to a .tsv file with elan annotations")
parser.add_argument("--elan_stats_fp", default=os.path.join(SAVE_TO, 'additional_stats_from_elan.json'), type=str,
                    help="Path to a json file with elan mean pos start and mean statistics")
parser.add_argument("--targets", default=[
    'pose_Rx',
    'perp_dist39_42_3d_pred_MLP_best_model_diff',
    'perp_dist39_42_3d_pred_MLP_best_model_no_features_diff',
], type=list, nargs='+', help="Targets to transpose")
parser.add_argument("--save_to", default=SAVE_TO, type=str,
                    help="Save path for OpenFace output with corrected distances")


def transpose_openface(openface_fp, elan_fp, elan_stats_fp, targets, save_to):
    df = pd.read_csv(openface_fp, index_col=[0, 1])
    elan_stats = json.load(open(elan_stats_fp))
    elan = pd.read_csv(elan_fp, sep='\t')
    mean_duration = int(np.mean(list(elan_stats['mean_duration'].values())))
    df = combine_w_elan(df, elan, elan_stats, mean_duration)
    df_transposed = norm_ix_transpose(df, targets, mean_duration)
    df_transposed.to_csv(os.path.join(save_to, 'open_face_transposed.csv'))


def _get_mean_pos_frame(d):
    pos_dict = defaultdict(list)
    for stype, values in d.items():
        for k, v in values.items():
            pos_dict[k].append(v)
    pos_dict = {k: np.mean(vs) for k, vs in pos_dict.items()}
    return pos_dict


def combine_w_elan(df, elan, elan_stats, mean_duration):
    start_pos_dict = _get_mean_pos_frame(elan_stats['pos_start'])
    end_pos_dict = _get_mean_pos_frame(elan_stats['pos_end'])
    elan.apply(_combine_by_row,
               axis=1,
               df=df,
               columns=[CATEGORICAL.BROWS],
               start_pos_dict=start_pos_dict,
               end_pos_dict=end_pos_dict)

    max_frame = df.reset_index().groupby([CATEGORICAL.VIDEO_NAME])[CATEGORICAL.FRAME].max()
    min_frame = df.reset_index().groupby([CATEGORICAL.VIDEO_NAME])[CATEGORICAL.FRAME].min()

    for video in max_frame.index:
        for i, boundary in enumerate([max_frame, min_frame]):
            ix = df.loc[(video, boundary[video]), INDEX.NORM]
            df.loc[(video, boundary[video]), INDEX.NORM] = [mean_duration, 0][i] if pd.isna(ix) else ix

    df[INDEX.NORM].interpolate('linear', inplace=True)
    df[INDEX.NORM] = df[INDEX.NORM].round()
    return df


def _combine_by_row(row, df, columns, start_pos_dict, end_pos_dict):
    pos = row[CATEGORICAL.POS]
    video_name = row[CATEGORICAL.VIDEO_NAME]
    if pd.notna(pos) and pos != 'Q':
        df.loc[(video_name, row['start_frames']), INDEX.NORM] = start_pos_dict[pos].round()
        df.loc[(video_name,
                min(row['end_frames'], df.loc[video_name].index.max())),
               INDEX.NORM] = end_pos_dict[pos].round()

    for column in columns:
        value = row[column]
        if value is None:
            continue
        df.loc[(video_name, slice(row['start_frames'], row['end_frames'])), column] = value


def norm_ix_transpose(df, targets, mean_duration):
    indexes = [['outer', 'inner'], targets, df.index.levels[0].tolist()]
    multi_indexes = pd.MultiIndex.from_product(indexes, names=['brows', 'metric', 'video_name'])

    transposed = pd.DataFrame(columns=np.arange(0, mean_duration + 1), index=multi_indexes, dtype=np.float64)
    poses = []
    for i in range(len(targets) - 1, -1, -1):
        if 'pose' in targets[i]:
            poses.append(targets.pop(i))
    for video_name in tqdm(df.index.levels[0]):
        _transpose_in_batches(df, video_name, targets, poses, transposed)
    transposed = meanfill(transposed, mean_duration)
    transposed = _make_cat_features(transposed)
    transposed.columns = transposed.columns.map(str)
    return transposed


def _make_cat_features(transposed):
    values = transposed.reset_index(level=CATEGORICAL.VIDEO_NAME)[CATEGORICAL.VIDEO_NAME].str.split("-",
                                                                                                    expand=True).values
    transposed[[CATEGORICAL.STYPE,
                CATEGORICAL.SENTENCE,
                CATEGORICAL.SPEAKER]] = values
    transposed['deaf'] = transposed.apply(lambda x: x[CATEGORICAL.SPEAKER] in SPEAKERS.DEAF, axis=1)
    return transposed


def norm_ix(df):
    df[INDEX.NORM].interpolate('linear', inplace=True)
    df[INDEX.NORM] = df[INDEX.NORM].round()


def meanfill(df, mean_duration):
    df_numeric = df.iloc[:, 0:mean_duration + 1]
    df_ffill = df_numeric.ffill(axis=1)
    df_bfill = df_numeric.bfill(axis=1)
    df_meanfill = (df_ffill + df_bfill) / 2
    df_meanfill = df_meanfill.ffill(axis=1).bfill(axis=1)
    df.iloc[:, 0:mean_duration + 1] = df_meanfill
    return df


def _transpose_in_batches(df, video_name, metrics, poses, transposed):
    index = df.loc[(video_name), INDEX.NORM]

    if transposed.loc[(['inner', 'outer'], metrics, video_name), index].isna().all(axis=None):
        for brows in ['inner', 'outer']:
            transposed.loc[(brows,
                            metrics,
                            video_name),
                           index] = df.loc[video_name,
                                           [f'{brows}_{metric}' for metric in metrics]].values.T
            transposed.loc[(brows, poses, video_name), index] = df.loc[video_name, poses].values.T
    else:
        for brows in ['inner', 'outer']:
            transposed.loc[(brows,
                            metrics,
                            video_name),
                           index] = np.mean([df.loc[video_name,
                                                    [f'{brows}_{metric}' for metric in metrics]].values.T,
                                             transposed.loc[(brows, metrics, video_name), index]].values, axis=0)
            transposed.loc[(brows,
                            poses,
                            video_name),
                           index] = np.mean([df.loc[video_name, poses].values.T,
                                             transposed.loc[(brows, poses, video_name).values, index]], axis=0)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    transpose_openface(args.openface_fp, args.elan_fp, args.elan_stats_fp, args.targets, save_to=args.save_to)
