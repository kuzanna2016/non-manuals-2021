import os
import pandas as pd
import numpy as np
import pickle
import cv2

from .extractors import proper_name
from .const import rate_to_delay, CATEGORICAL, STYPE, POS_FP

class Elan:

    def __init__(self,
                 elan,
                 meta_video,
                 save_to,
                 preprocessed=False,
                 ):
        self.elan = elan
        self.meta_video = meta_video
        self.save_to = save_to
        if not preprocessed:
            self._preprocess_elan()
        self._set_means()
        self.filtered_videos = self.elan[CATEGORICAL.VIDEO_NAME].unique()
        self.st_no_brows = self._find_st_no_brows()

    @classmethod
    def from_csv(cls, fp, save_to):
        elan = pd.read_csv(fp + 'elan.tsv', sep='\t', index_col=0)
        meta_video = pd.read_csv(fp + 'meta_video.tsv', sep='\t', index_col=0)
        return cls(elan=elan, save_to=save_to, meta_video=meta_video, preprocessed=True)

    def save_elan(self, fp):
        self.elan.to_csv(fp + 'elan.tsv', sep='\t')
        self.meta_video.to_csv(fp + 'meta_video.tsv', sep='\t')

    @classmethod
    def from_scratch(cls, elan_fp, save_to, video_fp=None, meta_video_fp=None):
        elan = pd.read_csv(elan_fp, sep='\t')
        if meta_video_fp is not None:
            meta_video = pd.read_csv(meta_video_fp, sep='\t')
        elif video_fp is None:
            raise ValueError('if no meta_video, video_fp must be given')
        else:
            meta_video = cls._get_meta_video(video_fp)
        return cls(elan=elan, meta_video=meta_video, save_to=save_to)

    def _preprocess_elan(self):
        # changing default column names and dropping useless columns
        self.elan['Файл'] = self.elan['Файл'].str.replace('\.eaf', '', regex=True)
        self.elan.rename(columns={'Время начала - миллисекунды': 'start_ms',
                             'Время окончания - миллисекунды': 'end_ms',
                             'Файл': CATEGORICAL.VIDEO_NAME},
                    inplace=True)
        self.elan.drop(columns=['Путь к файлу'], inplace=True)
        # making clean
        self.elan[CATEGORICAL.VIDEO_NAME] = self.elan[CATEGORICAL.VIDEO_NAME].apply(proper_name)
        self.elan[[CATEGORICAL.STYPE,
              CATEGORICAL.SENTENCE,
              CATEGORICAL.SPEAKER]] = self.elan[CATEGORICAL.VIDEO_NAME].str.split('-', expand=True)

        self._shift_elan()
        self._set_main_sign()
        self._drop_no_brow_no_sign_rows()
        self._filter_videos_on_number_of_signs()
        self._set_pos(pos_fp=POS_FP)

    def _set_means(self):
        self.dur_mean = (self.elan.groupby([CATEGORICAL.STYPE,
                                            CATEGORICAL.VIDEO_NAME])['frame_count'].max()
                         .groupby(CATEGORICAL.STYPE).mean().round())

        self.pos_start_mean = self.elan.groupby([CATEGORICAL.STYPE, CATEGORICAL.POS]).start_frames.mean().round()
        self.pos_end_mean = self.elan.groupby([CATEGORICAL.STYPE, CATEGORICAL.POS]).end_frames.mean().round()

    def _filter_videos_on_number_of_signs(self):
        no_wh = (self.elan[self.elan[CATEGORICAL.STYPE] != STYPE.WH.value]
                 .groupby([CATEGORICAL.VIDEO_NAME])
                 .filter(lambda x: x['main_sign'].nunique() == 2))
        wh = (self.elan[self.elan[CATEGORICAL.STYPE] == STYPE.WH.value]
              .groupby([CATEGORICAL.VIDEO_NAME])
              .filter(lambda x: x['main_sign'].nunique() == 3))
        self.elan = pd.concat([no_wh, wh])

    def _set_pos(self, pos_fp):
        with open(pos_fp, 'rb') as f:
            pos_dict = pickle.load(f)
        self.elan[CATEGORICAL.POS] = self.elan.apply(lambda x: pos_dict[x.sentence][x.main_sign] if type(x.main_sign) == str else None, axis=1)

    def _set_main_sign(self):
        def merge(row):
            right, left, signer, sentence = row['right-hand'], row['left-hand'], row[CATEGORICAL.SPEAKER], row[CATEGORICAL.SENTENCE]
            if signer == 'mira':
                if sentence == 'mama_ust' or sentence == 'dom_post':
                    sign = right
                else:
                    sign = left
            else:
                sign = right
            return sign

        self.elan['main_sign'] = self.elan.apply(merge, axis=1)

    def _drop_no_brow_no_sign_rows(self):
        self.elan.dropna(how='all', subset=['brows', 'main_sign'])

    @staticmethod
    def _get_meta_video(video_fp):
        meta_video = []
        for x in ['deaf', 'hearing']:
            path_sentence = video_fp + x + '/'
            sentences = os.listdir(path=path_sentence)
            for sentence in sentences:
                path_files = path_sentence + f'{sentence}/'
                files = os.listdir(path=path_files)
                for file in files:
                    if file.endswith('.mp4'):
                        path_video = path_files + file
                        vidcap = cv2.VideoCapture(path_video)
                        meta_video.append({CATEGORICAL.VIDEO_NAME: file[:-4], 'fps': vidcap.get(cv2.CAP_PROP_FPS),
                                           'frame_count': int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))})
        meta_video = pd.DataFrame(meta_video)
        # making clean video_name
        meta_video.video_name.apply(proper_name, inplace=True)
        return meta_video

    def _shift_elan(self):
        def ms_to_frames(row, meta_video, start='start'):
            video_name = row['video_name']

            fps = meta_video.fps[meta_video.video_name == video_name].values[0]
            frame_count = meta_video.frame_count[meta_video.video_name == video_name].values[0]

            duration = frame_count / fps
            delay = rate_to_delay(fps)
            spf = frame_count / (duration * 1000 - delay)
            return np.ceil((row[f'{start}_ms'] - delay) * spf)

        self.elan['start_frames'] = self.elan.apply(ms_to_frames, axis=1, meta_video=self.meta_video)
        self.elan['end_frames'] = self.elan.apply(ms_to_frames, axis=1, start='end', meta_video=self.meta_video)
        self.elan = self.elan.merge(self.meta_video[['frame_count', 'video_name']], on=['video_name'], how='left')

    def _find_st_no_brows(self):
        statements = self.elan[self.elan[CATEGORICAL.STYPE] == STYPE.ST.value]
        st_w_brows = statements.groupby(CATEGORICAL.VIDEO_NAME)['brows'].value_counts()
        st_w_brows = st_w_brows.reset_index(0)[CATEGORICAL.VIDEO_NAME].to_numpy()
        st_no_brows = np.setdiff1d(statements[CATEGORICAL.VIDEO_NAME].unique(), st_w_brows)
        return st_no_brows

    def _find_frames_no_brows(self):
        statements = self.elan[self.elan[CATEGORICAL.STYPE] == STYPE.ST.value]

        st_w_brows = statements.groupby(CATEGORICAL.VIDEO_NAME)['brows'].value_counts()
        st_w_brows = st_w_brows.reset_index(0)[CATEGORICAL.VIDEO_NAME].to_numpy()
        st_no_brows = np.setdiff1d(statements[CATEGORICAL.VIDEO_NAME].unique(), st_w_brows)
        return st_no_brows