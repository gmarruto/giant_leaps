from utils import giant_leaps_utils
import uvicorn
from fastapi import FastAPI, HTTPException, status
import asyncio

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


import utils.data_models as data_models
import api_parameters
import utils.pca_analysis as pca_analysis
import utils.ai_protein_screening as ai

api_params = api_parameters.load_parameters()

giantleaps_engine = giant_leaps_utils.connect_giantleaps_database()

app = FastAPI()


@app.get("/test")
async def test():
    return {"Everything OK"}

@app.post("/semantic_analysis")
def semantic_analysis():
    
    ai.semanticDataAnalysis(giantleaps_engine)

@app.post("/generate_clean_dataset_nutrients")
def generate_clean_dataset_nutrients():
    ai.generateCleanDatasetNutrients(giantleaps_engine)

@app.post("/generate_clean_dataset_aminoacids")
def generate_clean_dataset_aminoacids():
    ai.generateCleanDatasetAminoAcids(giantleaps_engine)

@app.get("/alternative_proteins_list", status_code=status.HTTP_200_OK)
def getAltProteinList(data: data_models.userProteinInput) -> data_models.ListAlernativeProtein:

    listRecommAtlProtein = ai.AltProtRecommendation(data, giantleaps_engine)

    return listRecommAtlProtein

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8009)