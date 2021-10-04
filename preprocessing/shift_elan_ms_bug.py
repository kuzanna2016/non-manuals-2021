import math
import pandas as pd


def main(elan_fp, meta_video_fp, save_to='saved/elan_preprocessed.tsv'):
    elan = pd.read_csv(elan_fp, sep='\t', index_col=0)
    meta_video = pd.read_csv(meta_video_fp, sep='\t')

    elan['start_frames'] = elan.apply(ms_to_frames, axis=1, meta_video=meta_video)
    elan['end_frames'] = elan.apply(ms_to_frames, axis=1, start='end', meta_video=meta_video)
    elan = elan.merge(meta_video[['frame_count', 'video_name']], on=['video_name'], how='left')
    elan.to_csv(save_to, sep='\t')


def ms_to_frames(row, meta_video, start='start'):
    video_name = row['video_name']

    fps = meta_video.fps[meta_video.video_name == video_name].values[0]
    frame_count = meta_video.frame_count[meta_video.video_name == video_name].values[0]

    duration = frame_count / fps
    delay = rate_to_delay(fps)
    spf = frame_count / (duration * 1000 - delay)
    return math.ceil((row[f'{start}_ms'] - delay) * spf)


def rate_to_delay(rate):
    if rate >= 30:
        return 133
    elif 30 > rate >= 28.22:
        return 141
    elif 28.22 > rate >= 28.14:
        return 142
    elif 28.14 > rate >= 23.09:
        return 173
    elif 23.09 > rate >= 22.88:
        return 174
    elif 22.88 > rate >= 22.81:
        return 175
    elif 22.81 > rate >= 19.54:
        return 204
    elif 19.54 > rate >= 19.24:
        return 207
    elif 19.24 > rate >= 19.22:
        return 208
    elif 19.22 > rate >= 16.58:
        return 241
    elif 16.58 > rate >= 14.73:
        return 271
    elif 14.73 > rate >= 14.68:
        return 272
    elif 14.68 > rate >= 14.60:
        return 273
    elif 14.60 > rate:
        return 274
