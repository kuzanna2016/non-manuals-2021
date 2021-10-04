import os
import cv2
import pandas as pd

from .extractors import proper_name
from .const import CATEGORICAL


def main(video_fp, save_to='saved/meta_video.tsv'):
    meta_video = []
    for (path, _, files) in os.walk(video_fp):
        for file in files:
            if file.endswith('.mp4'):
                path_video = os.path.join(path, file)
                vidcap = cv2.VideoCapture(path_video)
                meta_video.append({CATEGORICAL.VIDEO_NAME: file[:-4],
                                   'fps': vidcap.get(cv2.CAP_PROP_FPS),
                                   'frame_count': int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))})
    meta_video = pd.DataFrame(meta_video)
    meta_video[CATEGORICAL.VIDEO_NAME].apply(proper_name, inplace=True)
    meta_video.to_csv(save_to, sep='\t')
