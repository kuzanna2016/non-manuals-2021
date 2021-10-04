import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import random
import itertools
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
import json



from .elan import Elan
from .open_face import OpenFace
from bias_detection.model import Model
from .const import CATEGORICAL, BROWS, STYPE, PLOTS_FP, CV_FP


class Data:

    def __init__(self, of, elan, save_to, combined=False):
        self.of = of
        self.elan = elan
        self.save_to = save_to
        if not combined:
            self.of.combine_w_elan(elan)
        self.models = {}
        self.scaler = StandardScaler()

    @classmethod
    def from_scratch(cls, of_fp, elan_fp, video_fp, meta_video_fp, save_to, elan_from_scratch=True, of_from_scratch=True):
        if of_from_scratch:
            of = OpenFace.from_scratch(fp=of_fp, save_to=save_to)
        else:
            of = OpenFace.from_csv(fp=of_fp, save_to=save_to)
        if elan_from_scratch:
            elan = Elan.from_scratch(elan_fp=elan_fp, video_fp=video_fp, meta_video_fp=meta_video_fp, save_to=save_to)
            of.df = of.df.reindex(index=elan.filtered_videos, level=0)
        else:
            elan = Elan.from_csv(fp=elan_fp, save_to=save_to)
        return cls(of=of, elan=elan, save_to=save_to)

    @classmethod
    def from_csv(cls, fp, save_to):
        of = OpenFace.from_csv(fp, save_to)
        elan = Elan.from_csv(fp, save_to)
        return cls(of, elan, save_to, combined=True)

    def save(self, fp):
        self.elan.save_elan(fp)
        self.of.save(fp)

    def plot_video(self, video_name, metrics, brows_axes=1):
        if video_name not in self.of.df.index.levels[0]:
            return None
        fig, axes = plt.subplots(len(metrics), brows_axes, figsize=(8 * brows_axes, len(metrics) * 4))
        for i, metric in enumerate(metrics):
            if 'pose' not in metric:
                if brows_axes > 1:
                    for j, brow in enumerate(BROWS):
                        metric_name = f'{brow.value}_{metric}'
                        self.of.df.loc[(video_name), metric_name].plot(ax=axes[i][j], label=metric_name)
                else:
                    self.of.df.loc[(video_name), 'inner_' + metric].plot(ax=axes[i], label='inner')
                    self.of.df.loc[(video_name), 'outer_' + metric].plot(ax=axes[i], label='outer', linestyle='--')
            else:
                metric_name = metric
                if brows_axes > 1:
                    for j in range(brows_axes):
                        self.of.df.loc[(video_name), metric_name].plot(ax=axes[i][j], label=metric_name)
                        axes[i][j].invert_yaxis()
                else:
                    self.of.df.loc[(video_name), metric_name].plot(ax=axes[i], label=metric_name)
                    axes[i].invert_yaxis()
            if brows_axes > 1:
                axes[i][0].set_ylabel(metric)
            else:
                axes[i].set_ylabel(metric)

        video_mask = self.elan.elan[CATEGORICAL.VIDEO_NAME] == video_name
        video_mask = video_mask & self.elan.elan[CATEGORICAL.POS].notna()
        for (pos, start, end) in self.elan.elan.loc[(video_mask), [CATEGORICAL.POS,
                                                                   'start_frames',
                                                                   'end_frames']].values:
            for ax in axes.flatten():
                ax.axvline(start, alpha=0.4, color='red')
                ax.axvline(end, alpha=0.4, color='red')
                ax.text(np.mean([start, end]), ax.get_ylim()[0], pos,
                        horizontalalignment='center', verticalalignment='bottom',
                        fontsize='x-large', alpha=0.4, color='red')
        if not os.path.isdir(self.save_to + PLOTS_FP):
            os.mkdir(self.save_to + PLOTS_FP)
        plt.savefig(self.save_to + PLOTS_FP + f'{video_name}-{"_".join(metrics)}.png', layout='tight')

    def plot_metrics(self, metrics, x_axes=1, deaf=False):
        cmap = plt.get_cmap('Dark2')
        linestyles = ['-', '--', '-.']
        linewidth = 2.5

        if deaf:
            x_axes = 2
        fig, axes = plt.subplots(len(metrics), x_axes, figsize=(8 * x_axes, len(metrics) * 4))

        if deaf:
            axes[0][0].set_title('deaf', fontsize='x-large')
            axes[0][1].set_title('hearing', fontsize='x-large')
        elif x_axes > 1:
            for i, brow in enumerate(BROWS):
                axes[0][i].set_title(brow.value, fontsize='x-large')

        for n, stype in enumerate(STYPE):
            color = cmap(n)
            style = linestyles[n]

            mask = self.of.transposed[CATEGORICAL.STYPE] == stype.value
            for i, metric in enumerate(metrics):
                if 'pose' not in metric:
                    if x_axes > 1:
                        if deaf:
                            mask_deaf = mask & self.of.transposed['deaf']
                            mask_hear = mask & ~self.of.transposed['deaf']
                            for j, new_mask in enumerate([mask_deaf, mask_hear]):
                                for k, brow in enumerate(BROWS):
                                    subset = self.of.transposed.loc[(brow.value,
                                                                     metric,
                                                                     new_mask), '0.0':'70.0'].mean()
                                    subset.plot(ax=axes[i][j],
                                                color=color,
                                                linestyle=style,
                                                label=metric,
                                                linewidth=linewidth - k
                                                )
                        else:
                            for j, brow in enumerate(BROWS):
                                subset = self.of.transposed.loc[(brow.value,
                                                                 metric,
                                                                 mask), '0.0':'70.0'].mean()
                                subset.plot(ax=axes[i][j],
                                            color=color,
                                            linestyle=style,
                                            label=metric,
                                            linewidth=linewidth)
                    else:
                        self.of.transposed.loc[(BROWS.INNER.value,
                                                metric,
                                                mask), '0.0':'70.0'].mean().plot(ax=axes[i],
                                                                                 color=color,
                                                                                 label=BROWS.INNER.value,
                                                                                 linestyle=style,
                                                                                 linewidth=linewidth)
                        self.of.transposed.loc[(BROWS.OUTER.value,
                                                metric,
                                                mask), '0.0':'70.0'].mean().plot(ax=axes[i],
                                                                                 color=color,
                                                                                 label=BROWS.OUTER.value,
                                                                                 linestyle=style,
                                                                                 linewidth=linewidth - 1)
                else:
                    if x_axes > 1:
                        if deaf:
                            mask_deaf = mask & self.of.transposed['deaf']
                            mask_hear = mask & ~self.of.transposed['deaf']
                            for j, new_mask in enumerate([mask_deaf, mask_hear]):
                                self.of.transposed.loc[(BROWS.INNER.value,
                                                        metric,
                                                        new_mask), '0.0':'70.0'].mean().plot(ax=axes[i][j],
                                                                                             color=color,
                                                                                             linestyle=style,
                                                                                             label=metric,
                                                                                             linewidth=linewidth
                                                                                             )
                                axes[i][j].invert_yaxis()
                        else:
                            for j in range(x_axes):
                                self.of.transposed.loc[(BROWS.INNER.value,
                                                        metric,
                                                        mask), '0.0':'70.0'].mean().plot(ax=axes[i][j],
                                                                                         color=color,
                                                                                         linestyle=style,
                                                                                         label=metric,
                                                                                         linewidth=linewidth
                                                                                         )
                                axes[i][j].invert_yaxis()
                    else:
                        self.of.transposed.loc[(BROWS.INNER.value,
                                                metric,
                                                mask), '0.0':'70.0'].mean().plot(ax=axes[i],
                                                                                 color=color,
                                                                                 linestyle=style,
                                                                                 label=metric,
                                                                                 linewidth=linewidth
                                                                                 )
                        axes[i].invert_yaxis()

                if x_axes > 1:
                    axes[i][0].set_ylabel(metric)
                else:
                    axes[i].set_ylabel(metric)

        for pos, start, end in zip(self.elan.pos_start_mean.groupby(CATEGORICAL.POS).mean().round().index,
                                   self.elan.pos_start_mean.groupby(CATEGORICAL.POS).mean().round().values,
                                   self.elan.pos_end_mean.groupby(CATEGORICAL.POS).mean().round().values):
            if pos == 'Q':
                continue
            for ax in axes.flatten():
                ax.axvline(start, alpha=0.4, color='red')
                ax.axvline(end, alpha=0.4, color='red')
                ax.text(np.mean([start, end]), ax.get_ylim()[0], pos,
                        horizontalalignment='center', verticalalignment='bottom',
                        fontsize='x-large', alpha=0.4, color='red')

        fig.legend(plt.gca().lines[::2], ['st', 'polar_q', 'wh_q'], loc='center', bbox_to_anchor=(0.35, 0.05), ncol=3,
                   fontsize='x-large')
        if deaf or x_axes == 1:
            fig.legend(plt.gca().lines[:2], ['inner', 'outer'], loc='center', ncol=2, bbox_to_anchor=(0.65, 0.05),
                       fontsize='x-large')

        if not os.path.isdir(self.save_to + PLOTS_FP):
            os.mkdir(self.save_to + PLOTS_FP)
        plt.savefig(self.save_to + PLOTS_FP + f'stypes-{"_".join(metrics)}.png', layout='tight')

    def plot_logs(self):
        pass

    def prepare_data_for_regr(self, features, dummies, target, st_no_brows=True):
        self.of.make_dummies(dummies)
        features = features.copy()
        for dummy in dummies:
            features.extend(self.of.df[dummy].unique().tolist())
        self.of.make_trig_features(funcs=['cos'])

        if st_no_brows:
            X = self.of.df.loc[self.elan.st_no_brows, features]
            Y = self.of.df.loc[self.elan.st_no_brows, [brow.value + '_' + target for brow in BROWS]]
        else:
            self.of.df['brows_w_space'] = self.of.df.brows.fillna(method='bfill', limit=2)
            self.of.df['brows_w_space'] = self.of.df.brows_w_space.fillna(method='ffill', limit=2)
            st_frames_no_brows = (self.of.df[CATEGORICAL.STYPE] == STYPE.ST.value) & (self.of.df.brows_w_space.isna())
            X = self.of.df.loc[st_frames_no_brows, features]
            Y = self.of.df.loc[st_frames_no_brows, [brow.value + '_' + target for brow in BROWS]]
        return X, Y

    def cross_validate_model(self, X, Y, model, params, metrics=('rmse', 'mae', 'mse', 'mrae'), n_folds=4, test_size=0.25, random_state=0, st_no_brows=True):
        cv_logs = []
        if os.path.exists(self.save_to + CV_FP + 'cv_logs.json'):
            with open(self.save_to + CV_FP + 'cv_logs.json', 'r') as f:
                prev_logs = json.load(f)
        else:
            prev_logs = []

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
            cv = {'model': model['name'],
                  'target': Y.columns.tolist(),
                  'params': param_set,
                  'train': {'folds': [], 'mean_score': {}},
                  'val': {'folds': [], 'mean_score': {}},
                  'features': X.columns.tolist(),
                  'st_no_brows': st_no_brows,
                  }

            prev_models = [prev_model for prev_model in prev_logs if prev_model['model'] == model['name']]
            already_in_cv = False
            for prev_model in prev_models:
                if prev_model['target'] == Y.columns.tolist():
                    if all(key in prev_model['val']['mean_score'] for key in metrics):
                        if all(param_set.get(p) == v for p, v in prev_model['params'].items()):
                            already_in_cv = True
                            print('already in cv_logs')
                            break
            if already_in_cv:
                continue

            for train_index, test_index in ss.split(X):
                X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[
                    test_index]

                if model.get('kwargs') is None:
                    model['kwargs'] = {}
                regr = self.load_model(kind=model['name'], params=param_set, **model['kwargs'])
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
                json.dump(cv_logs, open(self.save_to + CV_FP + 'cv_logs_intermediate.json', 'w'))


        prev_logs.extend(cv_logs)
        with open(self.save_to + CV_FP + 'cv_logs.json', 'w') as f:
            json.dump(prev_logs, f)
        self.choose_best_model(cv_logs, metrics=metrics)
        self.choose_best_model(cv_logs, metrics=('mrae','rmse'), mean_metric=True)

    @staticmethod
    def choose_best_model(cv_logs, metrics=None, mean_metric=False):
        if not cv_logs:
            return None
        if metrics is None:
            metrics = cv_logs[0]['val']['mean_score'].keys()
        if mean_metric:
            print('\tmean', ' and '.join(metrics))

            metric_scores = [rec for rec in cv_logs if all(metric in rec['val']['mean_score'] for metric in metrics)]
            top = sorted(enumerate(metric_scores),
                         key=lambda x: sum(x[1]['val']['mean_score'][metric] for metric in metrics)/len(metrics))
            for n, t in top[:5]:
                print(t['target'])
                print(t['params'])
                if 'features' in t:
                    print(t['features'])
                print('mean', sum(t['val']['mean_score'][metric] for metric in metrics)/len(metrics))
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

    def load_model(self, kind='mlp', params=None, regularizer=None, name=None, keep_model=False):
        if kind == 'mlp':
            m = Model.mlp(params or {}, name=name, save_to=self.save_to)
        elif kind == 'lasso':
            m = Model.lr(params, regularizer='lasso', name=name, save_to=self.save_to)
        elif kind == 'ridge':
            m = Model.lr(params, regularizer='ridge', name=name, save_to=self.save_to)
        else:
            raise ValueError('no such kind of model')
        if keep_model:
            self.models[m.name] = m
        return m

    def fit_predict(self, X_train, y_train, model, params, metrics=('rmse', 'mae', 'mse', 'mrae'), st_no_brows=True, save_weights=True):
        if os.path.isfile(self.save_to + 'pred_logs.json'):
            with open(self.save_to + 'pred_logs.json', 'r') as f:
                prev_logs = json.load(f)
        else:
            prev_logs = []

        if model.get('kwargs') is None:
            model['kwargs'] = {}

        logs = {'model': model['name'],
                'target': y_train.columns.tolist(),
                'params': params,
                'features': X_train.columns.tolist(),
                'st_no_brows': st_no_brows,
                }

        regr = self.load_model(kind=model['name'], params=params, **model['kwargs'])
        regr.train(X_train, y_train, X_train, y_train, metrics)
        logs['train'] = regr.logs['train']

        prev_logs.append(logs)
        with open(self.save_to + 'pred_logs.json', 'w') as f:
            json.dump(prev_logs, f)

        new_metrics = set()
        y_pred = regr.predict(self.of.df.loc[:, X_train.columns.tolist()])
        for i, y_name in enumerate(y_train.columns.tolist()):
            name = y_name + '_pred_' + model['name']
            if name in self.of.df.columns:
                name += '_' + str(len(prev_logs))
            self.of.df[name] = y_pred[:, i]
            diff_name = name + '_diff'
            self.of.df[diff_name] = self.of.df[y_name] - self.of.df[name]
            for brow in BROWS:
                name = name.replace(brow.value + '_', '')
                diff_name = diff_name.replace(brow.value + '_', '')
            new_metrics.add(name)
            new_metrics.add(diff_name)

        if save_weights:
            regr.save_model()
        return list(new_metrics)

    def plot_faces(self, sentence):
        video = self.of.df.loc[sentence]
        video = video[video['confidence'] >= 0.8]
        Xs = video.filter(regex='^X_\d?\d').iloc[:, pd.np.r_[17:27, 36:48]]
        Ys = video.filter(regex='^Y_\d?\d').iloc[:, pd.np.r_[17:27, 36:48]]
        Zs = video.filter(regex='^Z_\d?\d').iloc[:, pd.np.r_[17:27, 36:48]]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('hot')
        n_frames = video.shape[0]
        for i, frame in enumerate(zip(Xs.values,Ys.values,Zs.values)):
            color = cmap(i/n_frames)
            frame = list(frame)
            frame[2] = frame[2] + i*50
            ax.scatter(*frame, color=color)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(sentence)
        ax.view_init(elev=10., azim=(0))

        f_name = self.save_to + f'{sentence}_eyebrow_movement'
        i = 1
        while os.path.isfile(f_name + '_' + str(i) + '.png'):
            i += 1
        plt.savefig(f_name + '_' + str(i) + '.png', bbox_inches='tight')


    def plot_samples(self,
                     models,
                     n_samples=5,
                     brow=BROWS.INNER.value,
                     target='perp_dist39_42_3d',
                     zero_line=False,
                     plot_pose=False):
        samples = random.sample(self.of.df.index.levels[0].tolist(), n_samples)
        n_cols = len(models) + 1 if plot_pose else len(models)
        fig, axes = plt.subplots(len(samples), n_cols, figsize=(15, 20))
        for n, sample in enumerate(samples):
            for j, model in enumerate(models):
                name = f'{brow}_{target}_{model}' if model else f'{brow}_{target}'
                axes[n][j].plot(self.of.df.loc[sample, name])
                axes[0][j].set_title(model)
            if plot_pose:
                name = 'pose_Rx'
                axes[n][j+1].plot(self.of.df.loc[sample, name])
                axes[n][j+1].invert_yaxis()
                axes[0][j+1].set_title(name)
            axes[n][0].set_ylabel(sample)

            mask = self.elan.elan.video_name == sample
            if self.elan.elan[mask].brows.notna().any():
                br_moves = self.elan.elan[mask].dropna(subset=['brows'])[
                    ['start_frames', 'end_frames']].values
                for k, (start, end) in enumerate(br_moves):
                    for i in range(n_cols):
                        axes[n][i].axvline(start, alpha=0.5, color='red')
                        axes[n][i].axvline(end, alpha=0.5, color='red')
                        axes[n][i].text(x=start + (end - start) / 2,
                                        y=axes[n][i].get_ylim()[0],
                                        s=self.elan.elan[mask]['brows'].dropna().iloc[k],
                                        horizontalalignment='center', verticalalignment='bottom',
                                        color='red',
                                        fontsize=10,
                                        alpha=0.5,
                                        )
        for ax in axes.flatten():
            ax.set_xlabel('')
            ax.tick_params(reset=True)
            if zero_line:
                ax.axhline(0, alpha=0.5, color='green')

        f_name = self.save_to + PLOTS_FP + f'{n_samples}_samples_compare_changes'
        i = 1
        while os.path.isfile(f_name + '_' + str(i) + '.png'):
            i += 1
        plt.savefig(f_name + '_' + str(i) + '.png', bbox_inches='tight')
