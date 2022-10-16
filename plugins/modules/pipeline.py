import logging
import os
from datetime import datetime

import dill
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


path = os.environ.get('PROJECT_PATH', '.')


def drop(df, columns):
    return df.drop(columns, axis=1)


def remove_outliers(df, column):
    q25 = df[column].quantile(0.25)
    q75 = df[column].quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
    df.loc[df[column] < boundaries[0], column] = round(boundaries[0])
    df.loc[df[column] > boundaries[1], column] = round(boundaries[1])

    return df


def add_features(df):
    import pandas as pd

    df['short_model'] = df['model'].apply(
        lambda x: x.lower().split(' ')[0] if not pd.isna(x) else x
    )
    df['age_category'] = df['year'].apply(
        lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average')
    )

    return df


def log(df):
    return df


def pipeline() -> None:
    df = pd.read_csv(f'{path}/data/train/homework.csv')

    X = df.drop('price_category', axis=1)
    y = df['price_category']

    preprocessor = Pipeline(
        [
            (
                'drop',
                FunctionTransformer(
                    drop,
                    kw_args={
                        'columns': [
                            'id',
                            'url',
                            'region',
                            'region_url',
                            'price',
                            'manufacturer',
                            'image_url',
                            'description',
                            'posting_date',
                            'lat',
                            'long',
                        ]
                    },
                ),
            ),
            (
                'remove_outliers',
                FunctionTransformer(remove_outliers, kw_args={'column': 'year'}),
            ),
            ('add_features', FunctionTransformer(add_features)),
            ('drop1', FunctionTransformer(drop, kw_args={'columns': ['model']})),
            (
                'column_transformer',
                ColumnTransformer(
                    transformers=[
                        (
                            'numerical',
                            Pipeline(
                                steps=[
                                    ('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler()),
                                ]
                            ),
                            make_column_selector(dtype_include=['int64', 'float64']),
                        ),
                        (
                            'categorical',
                            Pipeline(
                                steps=[
                                    (
                                        'imputer',
                                        SimpleImputer(strategy='most_frequent'),
                                    ),
                                    ('encoder', OneHotEncoder(handle_unknown='ignore')),
                                ]
                            ),
                            make_column_selector(dtype_include=object),
                        ),
                    ]
                ),
            ),
            ('log', FunctionTransformer(log)),
        ]
    )

    models = [LogisticRegression(solver='liblinear'), RandomForestClassifier(), SVC()]

    best_score = 0.0
    best_pipe = None
    for model in models:

        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])

        logging.info(
            f'model: {type(model).__name__}\n',
            '*' * 50,
        )

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        logging.info(
            f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}'
        )
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    logging.info(
        f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}'
    )

    best_pipe.fit(X, y)
    model_filename = (
        f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'
    )

    with open(model_filename, 'wb') as file:
        dill.dump(best_pipe, file)

    logging.info(f'Model is saved as {model_filename}')


if __name__ == '__main__':
    pipeline()
