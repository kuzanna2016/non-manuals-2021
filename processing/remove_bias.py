import os
import json
import pandas as pd
import argparse
import datetime

from const import BROWS
from model import Model
from const import CATEGORICAL, MODELS_FP, SAVE_TO, CONFIGS_FP, STYPE, FEATURES
from tools import make_dummies, make_trig_features

parser = argparse.ArgumentParser()
parser.add_argument("--openface_fp", default=os.path.join(SAVE_TO, 'open_face_with_distances.csv'), type=str,
                    help="Path to a .csv file with OpenFace outputs with computed distances")
parser.add_argument("--elan_stats_fp", default=os.path.join(SAVE_TO, 'additional_stats_from_elan.json'), type=str,
                    help="Path to a json file with computed frames with brows")
parser.add_argument("--sentence_level", default=True, type=bool,
                    help="If True only sentences with no brows movement will be used for training, if False, all frames without brows movement will be used")
parser.add_argument("--configs_fp", default=os.path.join(CONFIGS_FP, 'models.json'), type=str,
                    help="Path to a .json file with the models parameters")
parser.add_argument("--save_to", default=SAVE_TO, type=str,
                    help="Save path for OpenFace output with corrected distances")


def prepare_data_for_regr(df, features, dummies, target):
    df = make_trig_features(df, funcs=features)
    df = make_dummies(df, dummies)

    features = features.copy()
    for dummy in dummies:
        features.extend(df[dummy].unique().tolist())

    X = df.loc[:, features]
    Y = df.loc[:, [brow.value + '_' + target for brow in BROWS]]
    return X, Y


def filter_frames_w_brows(df, elan_stats):
    frames_w_brows = elan_stats['frames_w_brows']
    frames_no_brows = df.index.difference(frames_w_brows)
    mask = df.index.isin(frames_no_brows) & (df[CATEGORICAL.STYPE] == STYPE.ST.value)
    return mask


def filter_sentences_w_brows(df, elan_stats):
    videos_no_brows = elan_stats['videos_no_brows']
    mask = df.index.isin(videos_no_brows, level=0) & (df[CATEGORICAL.STYPE] == STYPE.ST.value)
    return mask


def fit_predict(
        X_train,
        Y_train,
        X_test,
        df,
        model,
        params,
        save_to,
        name=None,
        kwargs={}
):
    regr = Model.get_model(kind=model, params=params, name=name, save_to=save_to, **kwargs)
    regr.train(X_train, Y_train)

    y_pred = regr.predict(X_test)
    for i, y_name in enumerate(Y_train.columns.tolist()):
        column_name = y_name + '_pred_' + name if name is not None else model
        if column_name in df.columns:
            n = len([col for col in df.columns if column_name in col])
            column_name += '_' + str(n)
        df[column_name] = y_pred[:, i]
        diff_name = column_name + '_diff'
        df[diff_name] = df[y_name] - df[column_name]
    if save_to is not None:
        regr.save_model()
    return df


def remove_bias(config_fp, openface_fp, elan_stats_fp, sentence_level, save_to):
    df = pd.read_csv(openface_fp, index_col=[0, 1])
    elan_stats = json.load(open(elan_stats_fp))
    if sentence_level:
        mask = filter_sentences_w_brows(df, elan_stats)
    else:
        mask = filter_frames_w_brows(df, elan_stats)

    configs = json.load(open(config_fp))
    for config in configs:
        name = config.get("name", datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        model = config.get("model_name")
        params = config.get("params", {})
        target = config.get("target")
        features = config.get("features", FEATURES)
        dummies = config.get("dummies", [CATEGORICAL.SPEAKER, CATEGORICAL.SENTENCE])
        kwargs = config.get("kwargs", {})
        X_test, Y = prepare_data_for_regr(df, features, dummies, target)
        X = X_test[mask]
        Y = Y[mask]

        df = fit_predict(
            X, Y, X_test, df,
            model=model,
            params=params,
            kwargs=kwargs,
            name=name,
            save_to=os.path.join(save_to, MODELS_FP),
        )
    df.to_csv(os.path.join(save_to, 'open_face_with_bias_correction.csv'))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    remove_bias(args.configs_fp, args.openface_fp, args.elan_stats_fp, save_to=args.save_to,
                sentence_level=args.sentence_level)
