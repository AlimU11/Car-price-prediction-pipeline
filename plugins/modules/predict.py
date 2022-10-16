import glob
import gzip
import logging
import os
import dill
import json
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')


def get_model():
    list_of_files = glob.glob(f'{path}/data/models/*.pkl')
    latest_file = max(list_of_files, key=os.path.getctime)

    with open(latest_file, 'rb') as file:
        return dill.load(file)


def predict():
    pipe = get_model()
    predictions = {}

    for file in glob.glob(f'{path}/data/test/*.json'):
        with open(file) as f:
            data = json.load(f)
            predictions.update({data['id']: pipe.predict(pd.DataFrame([data]))[0]})

    df = pd.DataFrame.from_dict(predictions, orient='index').reset_index()
    df.columns = ['id', 'predictions']
    df.to_csv(f'{path}/data/predictions/predictions.csv', index=False)


if __name__ == '__main__':
    predict()
