from enum import Enum

SAVE_TO = 'saved/'
CV_FP = 'cross-validation/'
PLOTS_FP = 'plots/'
RAW_FP = 'raw_files/'


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
    NORM_IXS = 'norm_ix'



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

features = ['pose_Rx', 'cos_pose_Rx', 'pose_Tx',
            'pose_Ry', 'cos_pose_Ry', 'pose_Ty',
            'pose_Rz', 'cos_pose_Rz', 'pose_Tz']

# TODO: put metrics here and rearrange distance calculation

# TODO: make name gen function here
