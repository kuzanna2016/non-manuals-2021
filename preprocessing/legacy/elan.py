import os
import pandas as pd
import numpy as np
import json
import cv2
from collections import defaultdict
import argparse

from extractors import proper_name
from const import rate_to_delay, CATEGORICAL, STYPE, POS_DICT, RAW_FP, SAVE_TO

parser = argparse.ArgumentParser()
parser.add_argument("--elan_fp", default=os.path.join(RAW_FP, 'elan.tsv'), type=str,
                    help="Path to a .tsv Elan file with annotated videos")
parser.add_argument("--meta_fp", default=os.path.join(RAW_FP, 'meta_video.tsv'), type=str,
                    help="Path to a .tsv file with meta information about videos")
parser.add_argument("--videos_fp", default=None, type=list,
                    help="Path to a folder with videos, to get meta information, will be used recursively")
parser.add_argument("--save_to", default=SAVE_TO, type=str, help="Save path")


def custom_postproc(elan, save_to, **kwargs):
    elan = _set_main_sign(elan)
    elan = _drop_no_brow_no_sign_rows(elan)
    elan = _filter_videos_on_number_of_signs(elan)
    elan = _set_pos(elan)

    dur_mean, pos_start_mean, pos_end_mean = _set_means(elan)
    filtered_videos = elan[CATEGORICAL.VIDEO_NAME].unique()
    st_no_brows = _find_st_no_brows(elan)
    frames_w_brows = _find_frames_w_brows(elan)
    pos_dict = defaultdict(lambda: defaultdict(dict))
    for name, series in zip(['start', 'end'], [pos_start_mean, pos_end_mean]):
        for keys, value in series.to_dict().items():
            sentence, pos = keys
            pos_dict[f'pos_{name}'][sentence][pos] = value
    additional_info = {
        'mean_duration': dur_mean.to_dict(),
        'filtered_videos': filtered_videos.tolist(),
        'videos_no_brows': st_no_brows.tolist(),
        'frames_w_brows': frames_w_brows,
    }
    additional_info.update(pos_dict)
    json.dump(additional_info, open(os.path.join(save_to, 'additional_stats_from_elan.json'), 'w'))
    return elan


def _set_means(elan):
    dur_mean = (elan.groupby([CATEGORICAL.STYPE,
                              CATEGORICAL.VIDEO_NAME])['frame_count'].max()
                .groupby(CATEGORICAL.STYPE).mean().round())

    pos_start_mean = elan.groupby([CATEGORICAL.STYPE, CATEGORICAL.POS]).start_frames.mean().round()
    pos_end_mean = elan.groupby([CATEGORICAL.STYPE, CATEGORICAL.POS]).end_frames.mean().round()
    return dur_mean, pos_start_mean, pos_end_mean


def _filter_videos_on_number_of_signs(elan):
    no_wh = (elan[elan[CATEGORICAL.STYPE] != STYPE.WH.value]
             .groupby([CATEGORICAL.VIDEO_NAME])
             .filter(lambda x: x['main_sign'].nunique() == 2))
    wh = (elan[elan[CATEGORICAL.STYPE] == STYPE.WH.value]
          .groupby([CATEGORICAL.VIDEO_NAME])
          .filter(lambda x: x['main_sign'].nunique() == 3))
    elan = pd.concat([no_wh, wh])
    return elan


def _set_pos(elan):
    with open(os.path.join(RAW_FP, POS_DICT)) as f:
        pos_dict_text = f.read()
    pos_pairs = pos_dict_text.splitlines()
    pos_dict = {}
    for pair in pos_pairs:
        k, v = pair.split('\t')
        pos_dict[k] = v
    elan[CATEGORICAL.POS] = elan.apply(
        lambda x: pos_dict[x.main_sign] if type(x.main_sign) == str else None,
        axis=1
    )
    return elan


def _set_main_sign(elan):
    def merge(row):
        right, left, signer, sentence = row['right-hand'], row['left-hand'], row[CATEGORICAL.SPEAKER], row[
            CATEGORICAL.SENTENCE]
        if signer == 'mira':
            if sentence == 'mama_ust' or sentence == 'dom_post':
                sign = right
            else:
                sign = left
        else:
            sign = right
        return sign

    elan['main_sign'] = elan.apply(merge, axis=1)
    return elan


def _drop_no_brow_no_sign_rows(elan):
    elan.dropna(how='all', subset=['brows', 'main_sign'])
    return elan


def get_meta_video(videos_fp, _normalize_video_names_func):
    meta_video = []
    for fp, folders, files in os.walk(videos_fp):
        for file in files:
            if file.endswith('.mp4'):
                path_video = fp + file
                vidcap = cv2.VideoCapture(path_video)
                meta_video.append({CATEGORICAL.VIDEO_NAME: file[:-4], 'fps': vidcap.get(cv2.CAP_PROP_FPS),
                                   'frame_count': int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))})
    meta_video = pd.DataFrame(meta_video)
    # video names normalization
    meta_video.video_name.apply(_normalize_video_names_func, inplace=True)
    return meta_video


def shift_elan(elan, meta_video):
    def ms_to_frames(row, meta_video, start='start'):
        video_name = row['video_name']

        fps = meta_video.fps[meta_video.video_name == video_name].values[0]
        frame_count = meta_video.frame_count[meta_video.video_name == video_name].values[0]

        duration = frame_count / fps
        delay = rate_to_delay(fps)
        spf = frame_count / (duration * 1000 - delay)
        return np.ceil((row[f'{start}_ms'] - delay) * spf)

    elan['start_frames'] = elan.apply(ms_to_frames, axis=1, meta_video=meta_video)
    elan['end_frames'] = elan.apply(ms_to_frames, axis=1, start='end', meta_video=meta_video)
    elan = elan.merge(meta_video[['frame_count', 'video_name']], on=['video_name'], how='left')
    return elan


def _find_st_no_brows(elan):
    statements = elan[elan[CATEGORICAL.STYPE] == STYPE.ST.value]
    st_w_brows = statements.groupby(CATEGORICAL.VIDEO_NAME)['brows'].value_counts()
    st_w_brows = st_w_brows.reset_index(0)[CATEGORICAL.VIDEO_NAME].to_numpy()
    st_no_brows = np.setdiff1d(statements[CATEGORICAL.VIDEO_NAME].unique(), st_w_brows)
    return st_no_brows


def _find_frames_w_brows(elan):
    statements = elan[elan[CATEGORICAL.STYPE] == STYPE.ST.value]
    frames_w_brows = statements[statements.brows.notna()]
    frames_w_brows = frames_w_brows[['video_name', 'start_frames','end_frames']].to_dict('records')
    index = []
    for a in frames_w_brows:
        index.extend([(a['video_name'], i) for i in range(int(a['start_frames']), int(a['end_frames']) + 1)])
    return index


def preprocess_elan(elan_fp,
                    meta_fp=None,
                    videos_fp=None,
                    save_to='.',
                    _normalize_video_names_func=proper_name,
                    _custom_postprocessing=custom_postproc,
                    ):
    '''
    :param elan_fp: path to a .tsv Elan file with annotated videos
    :param meta_fp: path to a .tsv file with meta information about videos
    :param videos_fp: path or list of paths to a folder with videos, to get meta information
    :param save_to: path to a folder where to save the results
    :param _normalize_video_names_func: function that normalizes str representation of video name to stype-sentence-speaker format
    :return:
    '''
    elan = pd.read_csv(elan_fp, sep='\t')

    # changing default column names and dropping useless columns
    elan['Файл'] = elan['Файл'].str.replace('\.eaf', '', regex=True)
    elan.rename(columns={'Время начала - миллисекунды': 'start_ms',
                         'Время окончания - миллисекунды': 'end_ms',
                         'Файл': CATEGORICAL.VIDEO_NAME},
                inplace=True)
    elan.drop(columns=['Путь к файлу'], inplace=True)

    # video names normalization
    elan[CATEGORICAL.VIDEO_NAME] = elan[CATEGORICAL.VIDEO_NAME].apply(_normalize_video_names_func)
    elan[[CATEGORICAL.STYPE,
          CATEGORICAL.SENTENCE,
          CATEGORICAL.SPEAKER]] = elan[CATEGORICAL.VIDEO_NAME].str.split('-', expand=True)

    videos_meta = None
    if meta_fp is not None:
        videos_meta = pd.read_csv(meta_fp, sep='\t', index_col=0)
    elif videos_fp is not None:
        videos_meta = get_meta_video(videos_fp, _normalize_video_names_func)

    if videos_meta is not None:
        elan = shift_elan(elan, videos_meta)
        videos_meta.to_csv(os.path.join(save_to, 'videos_meta.tsv'), sep='\t')
    if _custom_postprocessing is not None:
        elan = _custom_postprocessing(elan, save_to)
    elan.to_csv(os.path.join(save_to, 'elan_preprocessed.tsv'), sep='\t', index=False)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    preprocess_elan(args.elan_fp,
                    meta_fp=args.meta_fp,
                    videos_fp=args.videos_fp,
                    save_to=args.save_to, )
