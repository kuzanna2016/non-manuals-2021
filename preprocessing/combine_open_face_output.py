import os
import pandas as pd
import argparse

from const import CATEGORICAL, RAW_FP, SAVE_TO
from extractors import proper_name

parser = argparse.ArgumentParser()
parser.add_argument("--open_face_fp", default=os.path.join(RAW_FP, 'of_output'), type=str,
                    help="Path to a folder with OpenFace output files")
parser.add_argument("--save_to", default=SAVE_TO, type=str, help="Save path")


def combine_of(fp, save_to):
    dfs = []
    names = []
    for fp, folders, files in os.walk(fp):
        for file in files:
            if file.endswith('.csv'):
                names.append(file[:-4])
                df = pd.read_csv(os.path.join(fp, file))
                dfs.append(df)
    df = pd.concat(dfs, keys=names, names=[CATEGORICAL.VIDEO_NAME])
    df = preprocess_of(df)
    df.to_csv(os.path.join(save_to, 'open_face_data_combined.csv'))


def preprocess_of(df, _normalize_video_names_func=proper_name):
    df.rename(columns=lambda col: col.replace(" ", ""), inplace=True)
    df.reset_index(inplace=True)
    df[CATEGORICAL.VIDEO_NAME] = df[CATEGORICAL.VIDEO_NAME].apply(_normalize_video_names_func)
    df[CATEGORICAL.FRAME] = df[CATEGORICAL.FRAME] - 1
    df[[CATEGORICAL.STYPE,
        CATEGORICAL.SENTENCE,
        CATEGORICAL.SPEAKER]] = df[CATEGORICAL.VIDEO_NAME].str.split('-', expand=True)
    df.set_index([CATEGORICAL.VIDEO_NAME, CATEGORICAL.FRAME], inplace=True)
    if 'level_1' in df.columns:
        df = df.drop(columns='level_1')
    df = df.sort_index(level=[0, 1])
    return df


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    combine_of(args.open_face_fp, save_to=args.save_to)
