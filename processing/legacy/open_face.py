import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
import argparse

from const import CATEGORICAL, INDEX, BROWS, SPEAKERS, RAW_FP, SAVE_TO
from extractors import proper_name
from distances import find_mean_dist, find_mean_perp_dist, find_mean_perp_plane_dist


class OpenFace:

    def __init__(self, df, save_to, preprocessed=False, transposed=None):
        self.df = df
        if not preprocessed:
            self.preprocess_of()
        self.df.sort_index(level=[0, 1], inplace=True)
        self.transposed = transposed
        self.save_to = save_to

    @classmethod
    def from_csv(cls, fp, save_to):
        df, transposed = cls.load(fp)
        return cls(df, save_to, preprocessed=True, transposed=transposed)

    @classmethod
    def from_scratch(cls, fp, save_to):
        df = cls.combine_of(fp)
        return cls(df, save_to)

    @staticmethod
    def combine_of(fp):
        dfs = []
        names = []
        for file in os.listdir(fp):
            if file.endswith('.csv'):
                names.append(file[:-4])
                df = pd.read_csv(fp + file)
                dfs.append(df)
        df = pd.concat(dfs, keys=names, names=[CATEGORICAL.VIDEO_NAME])
        return df

    def preprocess_of(self):
        self.df.rename(columns=lambda col: col.replace(" ", ""), inplace=True)
        self.df.reset_index(inplace=True)
        self.df[CATEGORICAL.VIDEO_NAME] = self.df[CATEGORICAL.VIDEO_NAME].apply(proper_name)
        self.df[CATEGORICAL.FRAME] = self.df[CATEGORICAL.FRAME] - 1
        self.df[[CATEGORICAL.STYPE,
                 CATEGORICAL.SENTENCE,
                 CATEGORICAL.SPEAKER]] = self.df[CATEGORICAL.VIDEO_NAME].str.split('-', expand=True)
        self.df.set_index([CATEGORICAL.VIDEO_NAME, CATEGORICAL.FRAME], inplace=True)

    def compute_distances(self, distances, override=False, plot_face=False):
        for dist, params in distances.items():
            for points in params:
                inner = points['inner']
                outer = points['outer']
                if 'perp' in dist:
                    if 'line' in dist:
                        find_mean_perp_dist(self.df,
                                            inner_brows=inner,
                                            outer_brows=outer,
                                            point_1=points['perp'][0],
                                            point_2=points['perp'][1],
                                            override=override)
                    else:
                        find_mean_perp_plane_dist(self.df,
                                                  inner_brows=inner,
                                                  outer_brows=outer,
                                                  perp_points=points['perp'],
                                                  override=override,
                                                  plot_face=plot_face)
                else:
                    find_mean_dist(self.df,
                                   inner_brows=inner,
                                   outer_brows=outer,
                                   point_2=points['point'][0],
                                   override=override)

    def save(self, fp='dev/raw files/'):
        self.df.to_csv(fp + 'open_face_old.csv', sep='\t')
        if self.transposed is not None:
            self.transposed.to_csv(fp + 'open_face_transposed.csv', sep='\t')

    @staticmethod
    def load(fp='/raw files/'):
        df = None
        transposed = None
        if os.path.isfile(fp + 'open_face.csv'):
            df = pd.read_csv(fp + 'open_face.csv', sep='\t', index_col=[0, 1])
        if os.path.isfile(fp + 'open_face_transposed.csv'):
            transposed = pd.read_csv(fp + 'open_face_transposed.csv', sep='\t', index_col=[0, 1, 2])

            # def f(x):
            #     try:
            #         return float(x)
            #     except:
            #         return x
            #
            # transposed.columns = transposed.columns.map(f)
        return df, transposed

    def combine_w_elan(self, elan):
        elan.elan.apply(self._combine_by_row,
                        axis=1,
                        columns=[CATEGORICAL.BROWS],
                        start_pos_dict=elan.pos_start_mean.groupby('pos').mean(),
                        end_pos_dict=elan.pos_end_mean.groupby('pos').mean())

        max_frame = self.df.reset_index().groupby([CATEGORICAL.VIDEO_NAME])[CATEGORICAL.FRAME].max()
        min_frame = self.df.reset_index().groupby([CATEGORICAL.VIDEO_NAME])[CATEGORICAL.FRAME].min()

        # TODO: mean frame from elan
        for video in max_frame.index:
            for i, boundary in enumerate([max_frame, min_frame]):
                ix = self.df.loc[(video, boundary[video]), INDEX.NORM]
                self.df.loc[(video, boundary[video]), INDEX.NORM] = [70, 0][i] if pd.isna(ix) else ix

        self.df[INDEX.NORM].interpolate('linear', inplace=True)
        self.df[INDEX.NORM] = self.df[INDEX.NORM].round()

    def _combine_by_row(self, row, columns, start_pos_dict, end_pos_dict):
        pos = row[CATEGORICAL.POS]
        video_name = row[CATEGORICAL.VIDEO_NAME]
        if pos is not None and pos != 'Q':
            self.df.loc[(video_name, row['start_frames']), INDEX.NORM] = start_pos_dict[pos].round()
            self.df.loc[(video_name,
                         min(row['end_frames'], self.df.loc[video_name].index.max())),
                        INDEX.NORM] = end_pos_dict[pos].round()

        for column in columns:
            value = row[column]
            if value is None:
                continue
            self.df.loc[(video_name, slice(row['start_frames'], row['end_frames'])), column] = value

    def norm_ix_transpose(self, metrics, override=False):
        if self.transposed is not None:
            if all(metric in self.transposed.index.levels[1] for metric in metrics):
                if not override:
                    print('all metrics are already transposed')
                    return None
            else:
                if not override:
                    metrics = [metric for metric in metrics if metric not in self.transposed.index.levels[1]]
                    if not metrics:
                        print('all metrics are already transposed ')
                        return None

        indexes = [['outer', 'inner'], metrics, self.df.index.levels[0].tolist()]
        multi_indexes = pd.MultiIndex.from_product(indexes, names=['brows', 'metric', 'video_name'])

        # TODO: take shape from elsewhere
        if self.transposed is None or override:
            self.transposed = pd.DataFrame(columns=np.arange(0, 71), index=multi_indexes, dtype=np.float64)
        poses = []
        for i in range(len(metrics) - 1, -1, -1):
            if 'pose' in metrics[i]:
                poses.append(metrics.pop(i))
        for video_name in tqdm(self.df.index.levels[0]):
            self._transpose_in_batches(video_name, metrics, poses, self.transposed)
        self.transposed = self.meanfill(self.transposed)
        self._make_cat_features()
        self.transposed.columns = self.transposed.columns.map(str)

    def _make_cat_features(self):
        values = self.transposed.reset_index(level=CATEGORICAL.VIDEO_NAME)[CATEGORICAL.VIDEO_NAME].str.split("-",
                                                                                                             expand=True).values
        self.transposed[[CATEGORICAL.STYPE,
                         CATEGORICAL.SENTENCE,
                         CATEGORICAL.SPEAKER]] = values
        self.transposed['deaf'] = self.transposed.apply(lambda x: x[CATEGORICAL.SPEAKER] in SPEAKERS.DEAF, axis=1)

    @staticmethod
    def meanfill(df):
        df_numeric = df.iloc[:, 0:71]
        df_ffill = df_numeric.ffill(axis=1)
        df_bfill = df_numeric.bfill(axis=1)
        df_meanfill = (df_ffill + df_bfill) / 2
        df_meanfill = df_meanfill.ffill(axis=1).bfill(axis=1)
        df.iloc[:, 0:71] = df_meanfill
        return df

    def _transpose_in_batches(self, video_name, metrics, poses, transposed):
        index = self.df.loc[(video_name), INDEX.NORM]

        if transposed.loc[(['inner', 'outer'], metrics, video_name), index].isna().all(axis=None):
            for brows in ['inner', 'outer']:
                transposed.loc[(brows,
                                metrics,
                                video_name),
                               index] = self.df.loc[video_name,
                                                    [f'{brows}_{metric}' for metric in metrics]].values.T
                transposed.loc[(brows, poses, video_name), index] = self.df.loc[video_name, poses].values.T
        else:
            for brows in ['inner', 'outer']:
                transposed.loc[(brows,
                                metrics,
                                video_name),
                               index] = np.mean([self.df.loc[video_name,
                                                             [f'{brows}_{metric}' for metric in metrics]].values.T,
                                                 transposed.loc[(brows, metrics, video_name), index]].values, axis=0)
                transposed.loc[(brows,
                                poses,
                                video_name),
                               index] = np.mean([self.df.loc[video_name, poses].values.T,
                                                 transposed.loc[(brows, poses, video_name).values, index]], axis=0)

    def make_dummies(self, columns):
        for column in columns:
            encoder = LabelBinarizer()
            transformed = encoder.fit_transform(self.df[column])
            for n, value in enumerate(self.df[column].unique()):
                self.df[value] = transformed[:, n]

    def make_trig_features(self, funcs):
        for axis in ['x', 'y', 'z']:
            for f in funcs:
                if f == 'sin':
                    func = np.sin
                elif f == 'cos':
                    func = np.cos
                elif f == 'tan':
                    func = np.tan
                else:
                    raise ValueError('no such function')
                self.df[f'{f}_pose_R{axis}'] = self.df[f'pose_R{axis}'].apply(func)
