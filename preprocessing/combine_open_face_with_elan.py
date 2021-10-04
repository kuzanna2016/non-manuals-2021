import pandas as pd

from .const import CATEGORICAL


def main(open_face_fp, elan_fp, save_to='saved/open_face_with_glosses.tsv'):
    of = pd.read_csv(open_face_fp, sep='\t', index_col=[0, 1])
    elan = pd.read_csv(elan_fp, sep='\t', index_col=0)

    filtered_videos = elan[CATEGORICAL.VIDEO_NAME].unique()
    of = of.reindex(index=filtered_videos, level=0)

    elan.apply(
        _combine_by_row,
        axis=1,
        of=of,
        columns=[CATEGORICAL.BROWS, CATEGORICAL.POS],
    )
    of.to_csv(save_to, sep='\t')


def _combine_by_row(row, of, columns):
    video_name = row[CATEGORICAL.VIDEO_NAME]
    for column in columns:
        value = row[column]
        if value is None:
            continue
        of.loc[(video_name, slice(row['start_frames'], row['end_frames'])), column] = value
