import pandas as pd

from .const import CATEGORICAL


def main(open_face_fp, elan_fp, save_to='saved/open_face_with_normalized_index.tsv'):
    of = pd.read_csv(open_face_fp, sep='\t', index_col=[0, 1])
    elan = pd.read_scv(elan_fp, sep='\t')

    dur_mean = elan.groupby([
        CATEGORICAL.STYPE,
        CATEGORICAL.VIDEO_NAME
    ])['frame_count'].max().groupby(CATEGORICAL.STYPE).mean().round()

    pos_start_mean = elan.groupby([CATEGORICAL.STYPE, CATEGORICAL.POS]).start_frames.mean().round()
    pos_end_mean = elan.groupby([CATEGORICAL.STYPE, CATEGORICAL.POS]).end_frames.mean().round()

    start_pos_dict = pos_start_mean.groupby('pos').mean().round()
    end_pos_dict = pos_end_mean.groupby('pos').mean().round()

    elan.apply(
        _combine_by_row,
        axis=1,
        of=of,
        start_pos_dict=start_pos_dict,
        end_pos_dict=end_pos_dict
    )

    max_frame = of.reset_index().groupby([CATEGORICAL.VIDEO_NAME])[CATEGORICAL.FRAME].max()
    min_frame = of.reset_index().groupby([CATEGORICAL.VIDEO_NAME])[CATEGORICAL.FRAME].min()

    for video in max_frame.index:
        for i, boundary in enumerate([max_frame, min_frame]):
            ix = of.loc[(video, boundary[video]), CATEGORICAL.NORM_IXS]
            of.loc[(video, boundary[video]), CATEGORICAL.NORM_IXS] = [dur_mean, 0][i] if pd.isna(ix) else ix

    of[CATEGORICAL.NORM_IXS].interpolate('linear', inplace=True)
    of[CATEGORICAL.NORM_IXS] = of[CATEGORICAL.NORM_IXS].round()
    of.to_csv(save_to, sep='\t')


def _combine_by_row(row, of, start_pos_dict, end_pos_dict):
    pos = row[CATEGORICAL.POS]
    video_name = row[CATEGORICAL.VIDEO_NAME]
    if pos is not None and pos != 'Q':
        start_frame = row['start_frames']
        of.loc[(video_name, start_frame), CATEGORICAL.NORM_IXS] = start_pos_dict[pos]
        end_frame = min(row['end_frames'], of.loc[video_name].index.max())
        of.loc[(video_name, end_frame), CATEGORICAL.NORM_IXS] = end_pos_dict[pos]
