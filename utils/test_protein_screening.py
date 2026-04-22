import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib

from fastapi import HTTPException, status

import giant_leaps_utils as giant_leaps_utils
import data_models as data_models
import protein_screening_parameters as protein_screening_parameters
import pca_analysis as pca_analysis

from sqlalchemy.sql import text

import re
from collections import Counter

import requests
import json

import datetime


#api_params = api_parameters.load_parameters()
protscreen_params = protein_screening_parameters.load_parameters()

giantleaps_engine = giant_leaps_utils.connect_giantleaps_database()


if __name__ == "__main__":

    selectRandomItems = "SELECT \
                                * \
                            FROM \
	                            dcf_data.food_sources fs2\
                            ORDER BY RANDOM() LIMIT 100\
                            "


    with giantleaps_engine.connect() as connection:
        trans = connection.begin()
        result_random_items = pd.read_sql_query(text(selectRandomItems), con=connection)
        trans.commit()
    
    df_random_tests = pd.DataFrame(result_random_items)

    print("Randomly selected items for testing:")
    df_test_results = pd.DataFrame(columns=['food_source_name', 'food_source_id', 'alternative_proteins'])
    for index, row in df_random_tests.iterrows():
        print(row['protein_source_or_food_product'], row['id'])

        url = "http://0.0.0.0:8009/alternative_proteins_list"

        payload = json.dumps({
        "proteinId": row['id'],
        })
        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        print(response.text)

        df_test_results = pd.concat([df_test_results, pd.DataFrame({
            'food_source_name': [row['protein_source_or_food_product']],
            'food_source_id': [row['id']],
            'alternative_proteins': [response.text]
        })], ignore_index=True)

    df_test_results.to_excel("/home/REMOVED/GIANT_LEAPS_DEV/tests/protein_screening_test_results_"+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".xlsx", index=False)