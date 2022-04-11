import numpy as np
from sklearn.preprocessing import LabelBinarizer


def make_dummies(df, columns):
    for column in columns:
        encoder = LabelBinarizer()
        transformed = encoder.fit_transform(df[column])
        for n, value in enumerate(df[column].unique()):
            df[value] = transformed[:, n]
    return df


def make_trig_features(df, funcs):
    for f in funcs:
        if f.endswith("_sin"):
            func = np.sin
        elif f.endswith("_cos"):
            func = np.cos
        elif f.endswith("_tan"):
            func = np.tan
        else:
            continue
        feature = f[:-4]
        df[f] = df[feature].apply(func)
    return df
