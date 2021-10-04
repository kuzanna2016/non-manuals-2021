import os
import pandas as pd

from .const import CATEGORICAL
from .extractors import proper_name


def main(fp, save_to='saved/open_face_combined.csv'):
    dfs = []
    names = []
    for file in os.listdir(fp):
        if file.endswith('.csv'):
            names.append(file[:-4])
            df = pd.read_csv(fp + file)
            dfs.append(df)
    df = pd.concat(dfs, keys=names, names=['video_name'])
    df = preproccess_open_face(df)
    df.to_csv(save_to, sep='\t')


def preproccess_open_face(df):
    df.rename(columns=lambda col: col.replace(" ", ""), inplace=True)
    df.reset_index(inplace=True)
    df[CATEGORICAL.VIDEO_NAME] = df[CATEGORICAL.VIDEO_NAME].apply(proper_name)
    df[CATEGORICAL.FRAME] = df[CATEGORICAL.FRAME] - 1
    df[[CATEGORICAL.STYPE,
        CATEGORICAL.SENTENCE,
        CATEGORICAL.SPEAKER]] = df[CATEGORICAL.VIDEO_NAME].str.split('-', expand=True)
    df.set_index([CATEGORICAL.VIDEO_NAME, CATEGORICAL.FRAME], inplace=True)
    df.sort_index(level=[0, 1], inplace=True)
    return df
