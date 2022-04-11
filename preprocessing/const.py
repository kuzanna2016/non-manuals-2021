from enum import Enum

POS_DICT='pos_dict.tsv'
SAVE_TO = 'saved/'
CV_FP = 'cross-validation/'
PLOTS_FP = 'plots/'
RAW_FP = 'raw files/'
CONFIGS = 'configs'
CV_LOGS = 'logs.json'
CV_LOGS_INTERMEDIATE = 'logs_intermediate.json'


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


class INDEX:
    NORM = 'norm_ix'


class CATEGORICAL:
    VIDEO_NAME = 'video_name'
    SPEAKER = 'speaker'
    SENTENCE = 'sentence'
    STYPE = 'sType'
    POS = 'pos'
    FRAME = 'frame'
    BROWS = 'brows'


class STYPE(Enum):
    WH = 'part_q'
    POLAR = 'gen_q'
    ST = 'st'

class BROWS(Enum):
    INNER = 'inner'
    OUTER = 'outer'

class SPEAKERS:
    DEAF = ['alt', 'kar', 'makh', 'mira', 'tanya']
    HEARING = []


class BASELINE:
    MLP = {'hidden_layer_sizes': [24, 20],
           'activation': 'logistic',
           'solver': 'lbfgs',
           'alpha': 1e-1,
           'max_iter': 1500,
           'tol': 1e-4,
           'learning_rate': 'adaptive'
    }
    LASSO = {'alpha': 1e-5,
             'tol': 1e-05,
             'max_iter': 10000}


class CV_BEST_MODEL:
    MLP = {'hidden_layer_sizes': [22, 28],
           'activation': 'logistic',
           'solver': 'lbfgs',
           'alpha': 1e-1,
           'max_iter': 1500,
           'tol': 1e-3,
    }

FEATURES = ['pose_Rx', 'pose_Rx_cos', 'pose_Tx',
            'pose_Ry', 'pose_Ry_cos', 'pose_Ty',
            'pose_Rz', 'pose_Rz_cos', 'pose_Tz']

# TODO: put metrics here and rearrange distance calculation
METRICS = ['rmse', 'mrae', 'mae', 'mse']
# TODO: make name gen function here
