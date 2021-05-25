import matplotlib.pyplot as plt
import numpy as np
import itertools
import json
import warnings


from .data import Data
from .model import Model
from .const import features, CATEGORICAL, SAVE_TO, CV_FP, RAW_FP


def main(of_fp=None, elan_fp=None, meta_video_fp=None, video_fp=None, save_to=SAVE_TO, raw_fp=RAW_FP):

    # load data =========================================================================

    # make data from scratch
    data = Data.from_scratch(of_fp=of_fp,
                             elan_fp=elan_fp,
                             video_fp=video_fp,
                             meta_video_fp=meta_video_fp,
                             save_to=save_to,
                             elan_from_scratch=True,
                             of_from_scratch=True)

    # load already made data
    # data = Data.from_csv(fp=raw_fp, save_to=save_to)


    # compute distances =================================================================

    # write config
    inner = (20,23)
    outer = (18,25)
    d = {'perp_plane':[
             {'inner': inner,
              'outer': outer,
              'perp': (27,)},
         ],
         'perp_line': [{'inner': inner,
                        'outer': outer,
                        'perp': (39, 42)}]
    }

    data.of.compute_distances(d)


    # perform cross-validation ===========================================================

    # write config
    models = [
        {
            'name': 'mlp',
            'params': {
                'hidden_layer_sizes': list(itertools.product(np.arange(18, 30, 2).tolist(),
                                                             np.arange(18, 30, 2).tolist())),
                'activation': ['logistic'],
                'solver': ['lbfgs'],
                'alpha': [1e-1],
                'max_iter': [1500],
                'tol': [1e-3]
            }
        },
        {
            'name': 'lasso',
            'params': {
                'alpha': [1e-5],
                'tol': [1e-3],
                'max_iter': [10000]
            },
        },
        {
            'name': 'ridge',
            'params': {
                'alpha': [1e-5],
                'tol': [1e-3],
                'max_iter': [10000]
            },
        }
    ]

    # # cross-validate
    for model in models:
        for target in ['perp_plane_dist27_3d']:
            X, Y = data.prepare_data_for_regr(features, [], target, st_no_brows=False)
            print(target)
            print(Y.aggregate(['mean', 'std']))
            print(data.of.df[features + Y.columns.tolist()].corr().iloc[:, 0])
            plot_corr(data.of.df[features + Y.columns.tolist()].corr(), features + Y.columns.tolist())
            data.cross_validate_model(X, Y,
                                      model=model,
                                      params=model['params'],
                                      st_no_brows=False
                                      )
    cv_logs = Model.load_logs(save_to, is_cv=True)
    data.choose_best_model(cv_logs, metrics=('mrae', 'rmse'))



    # get models predictions ================================================================


    # write config
    models = [
        {
            'name': 'lasso',
            'params': {
                'alpha': 1e-5,
                'tol': 1e-03,
                'max_iter': 10000
            },
            'st_no_brow': False,
        },
        {
            'name': 'mlp',
            'params': {
                'hidden_layer_sizes': [22, 28],
                'activation': 'logistic',
                'solver': 'lbfgs',
                'alpha': 1e-1,
                'max_iter': 1500,
                'tol': 1e-3,
            },
            'st_no_brow': False,
        },
    ]

    metrics = ['pose_Rx',
               'perp_dist39_42_3d']
    for model in models:
        for target in ['perp_dist39_42_3d']:
            X, Y = data.prepare_data_for_regr(features,
                                              [CATEGORICAL.SPEAKER, CATEGORICAL.SENTENCE],
                                              target,
                                              st_no_brows=model.get('st_no_brows') or True)
            print(target)
            print(Y.aggregate(['mean', 'std']))
            # plot_corr(data.of.df[features+Y.columns.tolist()].corr(), features+Y.columns.tolist())
            new_metrics = data.fit_predict(X_train=X,
                                           y_train=Y,
                                           model=model,
                                           params=model['params'],
                                           st_no_brows=False
                                          )
            metrics.extend(new_metrics)


    # transpose and save =====================================================================================
    data.of.norm_ix_transpose(metrics)
    data.save(save_to)


    # plot mean metrics ======================================================================================
    data.plot_metrics(metrics=[m for m in metrics if 'diff' in m or 'pose' in m])

    # plot one video
    data.plot_video('gen_q-dev_upala-vik',
                    metrics,
                    brows_axes=1,)

    # plot mean metrics with two axes for eyebrows
    data.plot_metrics(metrics,
                      x_axes=2,)

    # plot multiple videos
    data.plot_samples(models=[m[m.find('pred'):] for m in metrics if 'pred' in m] + [''], plot_pose=True)

    # plot faces
    for sentence in ['st-dom_post-tanya', 'gen_q-papa_beg-kar', 'part_q-dev_upala-vik']:
        data.plot_faces(sentence)

    # plot deaf and hearing differences
    data.plot_metrics(metrics,
                      deaf=True,)
    plt.show()


    # save sorted logs =============================================================================================
    cv_logs = Model.load_logs(save_to, is_cv=True)
    top = Data.choose_best_model(cv_logs, metrics=('mrae', 'rmse'), mean_metric=True)
    json.dump(top, open(save_to + CV_FP + 'cv_ranked.json', 'w'))


def plot_corr(cor_array, names):
    plt.matshow(cor_array, cmap=plt.cm.bwr)
    plt.gcf().set_size_inches(10, 5)
    plt.gca().set_aspect(0.3)
    plt.xticks(np.arange(0, len(names)), names, rotation=90)
    plt.yticks(np.arange(0, len(names)), names)
    plt.colorbar(orientation='horizontal')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main(of_fp='raw_files/of_output/',
         elan_fp='raw_files/raw_elan.txt',
         video_fp='../data/',
         meta_video_fp='raw_files/meta_video.tsv')
