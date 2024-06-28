import os
import re
import pandas as pd
import pickle
import numpy as np
import os
from inference import PMGInference

if __name__ == "__main__":
    DATA_PATH_MOVIE = './data/raw_data/ml-latest-small'
    item_data_path = os.path.join(DATA_PATH_MOVIE, 'movies.csv')
    mv_info = pd.read_csv(item_data_path)
    
    keywords_in_id = {}
    inference = PMGInference()
    for index, row in mv_info.iterrows():
        row['id'] = row['movieId']
        row['name'] = row['title']
        only_name = re.findall(r'^(.*) \(\d+\) *$', row['name'])
        if len(only_name):
            row['name'] = only_name[0]
        movie_id = str(row['id'])
        item_data = dict(row)
        keywords_in_id[row['id']] = inference.itemProcess('movie', item_data)
    
    with open('keywords_in_id.pkl', 'wb') as f:
        pickle.dump(keywords_in_id, f)

