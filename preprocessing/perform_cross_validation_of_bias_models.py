import random
import itertools
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import datetime

from sklearn.model_selection import ShuffleSplit

from const import BROWS
from model import Model
from const import CATEGORICAL, CV_FP, SAVE_TO, CONFIGS, STYPE, METRICS, FEATURES, CV_LOGS, CV_LOGS_INTERMEDIATE
from tools import make_dummies, make_trig_features

parser = argparse.ArgumentParser()
parser.add_argument("--openface_fp", default=os.path.join(SAVE_TO, 'open_face_with_distances.csv'), type=str,
                    help="Path to a .csv file with OpenFace outputs with computed distances")
parser.add_argument("--elan_stats_fp", default=os.path.join(SAVE_TO, 'additional_stats_from_elan.json'), type=str,
                    help="Path to a json file with computed frames with brows")
parser.add_argument("--sentence_level", default=True, type=bool,
                    help="If True only sentences with no brows movement will be used for cross validation, if False, all frames without brows movement will be used")
parser.add_argument("--config_fp", default=os.path.join(CONFIGS, 'cross_validation.json'), type=str,
                    help="Path to a .json file with cross-validation configs")
parser.add_argument("--save_to", default=os.path.join(SAVE_TO, CV_FP), type=str,
                    help="Save path for logs of cross-validation")


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
    filtered_df = df.filter(frames_no_brows, axis=0)
    filtered_df = filtered_df[filtered_df[CATEGORICAL.STYPE] == STYPE.ST.value]
    return filtered_df


def filter_sentences_w_brows(df, elan_stats):
    videos_no_brows = elan_stats['videos_no_brows']
    filtered_df = df[df.index.isin(videos_no_brows, level=0) & (df[CATEGORICAL.STYPE] == STYPE.ST.value)]
    return filtered_df


def cross_validate_model(
        X,
        Y,
        model,
        params,
        metrics,
        save_to,
        kwargs={},
        n_folds=4,
        test_size=0.25,
        random_state=0
):
    cv_logs = []
    ss = ShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=random_state)
    product = list(itertools.product(*params.values()))
    random.shuffle(product)

    print('Number of folds:', n_folds)
    print('Train size:', 0.75 * X.shape[0])
    print('Test size:', 0.25 * X.shape[0])
    print('Number of params combinations:', len(product))
    n = 0
    for param_set in tqdm(product):
        param_set = dict(zip(params.keys(), param_set))
        cv = {'model': model,
              'target': Y.columns.tolist(),
              'params': param_set,
              'train': {'folds': [], 'mean_score': {}},
              'val': {'folds': [], 'mean_score': {}},
              'features': X.columns.tolist(),
              }

        for train_index, test_index in ss.split(X):
            X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[
                test_index]
            regr = Model.get_model(kind=model, params=param_set, save_to=save_to, **kwargs)
            regr.train(X_train, y_train, X_test, y_test, metrics)

            cv['train']['folds'].append(regr.logs['train'])
            cv['val']['folds'].append(regr.logs['val'])

        for subset in ['train', 'val']:
            for m in cv[subset]['folds'][0].keys():
                m_sum = 0
                for score in cv[subset]['folds']:
                    if isinstance(score[m], list):
                        m_sum += np.mean(score[m])
                    else:
                        m_sum += score[m]
                cv[subset]['mean_score'][m] = m_sum / n_folds
        cv_logs.append(cv)
        n += 1

        if n % 10 == 0:
            json.dump(cv_logs, open(os.path.join(save_to, CV_LOGS_INTERMEDIATE), 'w'))
    return cv_logs


def choose_best_model(cv_logs, metrics=None, mean_metric=False):
    if not cv_logs:
        return None
    if metrics is None:
        metrics = cv_logs[0]['val']['mean_score'].keys()
    if mean_metric:
        print('\tmean', ' and '.join(metrics))

        metric_scores = [rec for rec in cv_logs if all(metric in rec['val']['mean_score'] for metric in metrics)]
        top = sorted(enumerate(metric_scores),
                     key=lambda x: sum(x[1]['val']['mean_score'][metric] for metric in metrics) / len(metrics))
        for n, t in top[:5]:
            print(t['target'])
            print(t['params'])
            if 'features' in t:
                print(t['features'])
            print('mean', sum(t['val']['mean_score'][metric] for metric in metrics) / len(metrics))
            for metric in t['val']['mean_score']:
                print(metric, '{:.5}'.format(t['val']['mean_score'][metric]))
    else:
        for metric in metrics:
            print('\t', metric)
            metric_scores = [rec for rec in cv_logs if metric in rec['val']['mean_score']]
            top = sorted(enumerate(metric_scores),
                         key=lambda x: x[1]['val']['mean_score'][metric])
            for n, t in top[:5]:
                print(t['target'])
                print(t['params'])
                print('mean {:.5}'.format(t['val']['mean_score'][metric]))
    return top


def perform_cross_validation(config_fp, openface_fp, elan_stats_fp, sentence_level, save_to):
    df = pd.read_csv(openface_fp, index_col=[0, 1])
    elan_stats = json.load(open(elan_stats_fp))
    if sentence_level:
        df = filter_sentences_w_brows(df, elan_stats)
    else:
        df = filter_frames_w_brows(df, elan_stats)

    configs = json.load(open(config_fp))
    for config in configs:
        name = config.get("name", datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        model = config.get("model_name")
        params = config.get("params", {})
        targets = config.get("targets")
        features = config.get("features", FEATURES)
        dummies = config.get("dummies", [CATEGORICAL.SPEAKER, CATEGORICAL.SENTENCE])
        metrics = config.get("metrics", METRICS)
        kwargs = config.get("kwargs", {})
        for target in targets:
            X, Y = prepare_data_for_regr(df, features, dummies, target)
            cv_logs = cross_validate_model(
                X, Y,
                model=model,
                params=params,
                kwargs=kwargs,
                metrics=metrics,
                save_to=save_to,
            )

            if os.path.exists(os.path.join(save_to, CV_LOGS)):
                with open(os.path.join(save_to, CV_LOGS), 'r') as f:
                    prev_logs = json.load(f)
            else:
                prev_logs = []

            prev_logs.extend(cv_logs)
            with open(os.path.join(save_to, CV_LOGS), 'w') as f:
                json.dump(prev_logs, f)
            top = choose_best_model(cv_logs, metrics=metrics)
            json.dump(top, open(os.path.join(save_to, f'{name}_{target}.json'), 'w'))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    perform_cross_validation(args.config_fp, args.openface_fp, args.elan_stats_fp, save_to=args.save_to,
                             sentence_level=args.sentence_level)
