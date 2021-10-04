import pandas as pd

from preprocessing.extractors import proper_name
from preprocessing.const import CATEGORICAL, STYPE


def main(elan_fp, gloss_to_pos_fp, save_to='saved/elan_preproccessed.tsv'):
    elan = pd.read_csv(elan_fp, sep='\t')
    elan = _rename_elan_columns(elan)
    elan = _get_the_features_from_video_file_names(elan)
    elan = _set_main_sign_hand(elan)
    elan = _drop_no_brow_movement_no_sign_rows(elan)
    elan = _set_pos(elan, gloss_to_pos_fp)
    elan.save_to_csv(save_to, sep='\t')


def _rename_elan_columns(elan):
    elan['Файл'] = elan['Файл'].str.replace('\.eaf', '', regex=True)
    elan.rename(columns={'Время начала - миллисекунды': 'start_ms',
                         'Время окончания - миллисекунды': 'end_ms',
                         'Файл': CATEGORICAL.VIDEO_NAME},
                inplace=True)
    elan.drop(columns=['Путь к файлу'], inplace=True)
    return elan


def _get_the_features_from_video_file_names(elan):
    elan[CATEGORICAL.VIDEO_NAME] = elan[CATEGORICAL.VIDEO_NAME].apply(proper_name)
    elan[[CATEGORICAL.STYPE,
          CATEGORICAL.SENTENCE,
          CATEGORICAL.SPEAKER]] = elan[CATEGORICAL.VIDEO_NAME].str.split('-', expand=True)
    return elan


def _set_main_sign_hand(elan):
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


def _drop_no_brow_movement_no_sign_rows(elan):
    elan.dropna(how='all', subset=[CATEGORICAL.BROWS, 'main_sign'])
    return elan


def _filter_videos_on_number_of_signs(elan):
    polar_qs_and_statements = (elan[elan[CATEGORICAL.STYPE] != STYPE.WH.value]
                               .groupby([CATEGORICAL.VIDEO_NAME])
                               .filter(lambda x: x['main_sign'].nunique() == 2))
    wh_qs = (elan[elan[CATEGORICAL.STYPE] == STYPE.WH.value]
             .groupby([CATEGORICAL.VIDEO_NAME])
             .filter(lambda x: x['main_sign'].nunique() == 3))
    elan = pd.concat([polar_qs_and_statements, wh_qs])
    return elan


def _set_pos(elan, gloss_to_pos_fp):
    with open(gloss_to_pos_fp, encoding='utf-8') as f:
        pos_dict = f.read()
    pos_dict = [ln.split(',') for ln in pos_dict.splitlines()]
    pos_dict = {gloss: pos for (gloss, pos) in pos_dict}
    elan[CATEGORICAL.POS] = elan.apply(
        lambda x: pos_dict[x.main_sign] if type(x.main_sign) == str else None,
        axis=1
    )
    return elan
