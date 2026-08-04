
import sys

from matplotlib.pyplot import text
from numpy.compat import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sqlalchemy import engine
from sqlalchemy.sql import text
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import api_parameters
import giant_leaps_utils as giant_leaps_utils

import pandas as pd
import numpy as np

# Step 1: Implement k-NN
def euclidean_distance(x1, x2):
    return round(float(np.sqrt(np.sum((x1 - x2) ** 2))),4)

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]
        #most_common = Counter(k_nearest_labels).most_common(1)
        return k_nearest_labels, k_nearest_distances #most_common[0][0]

def compute_diversity(recommendations, catalog):
    diversities = []
    for recs in recommendations:
        if len(recs) < 2:
            continue
        vectors = np.array([list(catalog[item].values()) for item in recs])
        sim_matrix = cosine_similarity(vectors)
        upper_tri_indices = np.triu_indices(len(recs), k=1)
        similarities = sim_matrix[upper_tri_indices]
        diversity = 1 - np.mean(similarities)
        diversities.append(diversity)
    return float(np.mean(diversities)) if diversities else 0

def compute_individual_diversity(recommendations, catalog):
    vectors = np.array([list(catalog[item].values()) for item in recommendations])
    sim_matrix = cosine_similarity(vectors)
    upper_tri_indices = np.triu_indices(len(recommendations), k=1)
    similarities = sim_matrix[upper_tri_indices]
    diversity = 1 - np.mean(similarities)
    return float(diversity)

def make_model_candidates(random_state=42):
    # Base pipeline: scaled (already scaled during data treatment) + model
    pipe_knn_custom = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", KNN(k=10))
    ])

    # Pipeline with PCA + kNN custom
    pipe_pca75_knn_custom = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.75, svd_solver='full',random_state=random_state)),
        ("model", KNN(k=10))
    ])

    # Pipeline with PCA + kNN custom
    pipe_pca85_knn_custom = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.85, svd_solver='full',random_state=random_state)),
        ("model", KNN(k=10))
    ])

    # Pipeline with PCA + kNN custom
    pipe_pca90_knn_custom = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.90, svd_solver='full',random_state=random_state)),
        ("model", KNN(k=10))
    ])

    candidates = [
        ("knn_custom", pipe_knn_custom, {
            "model__k": [10]
        }),
        ("pipe_pca75_knn_custom", pipe_pca75_knn_custom, {
            "model__k": [10]
        }),
        ("pipe_pca85_knn_custom", pipe_pca85_knn_custom, {
            "model__k": [10]
        }),
        ("pipe_pca90_knn_custom", pipe_pca90_knn_custom, {
            "model__k": [10]
        })
    ]
    return candidates

def run_model_selection(X, y, scoring="f1_macro", n_jobs=-1, random_state=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results = {}
    for name, pipe, param_grid in make_model_candidates(random_state):
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=True
        )
        gs.fit(X, y)
        results[name] = gs
        print(f"[{name}] best_score={gs.best_score_:.4f} best_params={gs.best_params_}")
    return results


if __name__ == "__main__":

    api_params = api_parameters.load_parameters()

    giantleaps_engine = giant_leaps_utils.connect_giantleaps_database(api_params['db_conn_file'], api_params['db_name'])

    with giantleaps_engine.connect() as connection:
        trans = connection.begin()
        test_nutrients_data_model = pd.read_sql_query("SELECT\
                                                                psfg.id,\
                                                                psfg.category,\
                                                                ndim.*\
                                                            FROM\
                                                                public.nutrients_data_imputation_model_input ndim\
                                                            INNER JOIN protein_source_format_gm psfg ON ndim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                                            INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                                            WHERE\
                                                                pst.source_type = 'Traditional';" , con=connection)
        test_nutrients_aminoacids_data_model = pd.read_sql_query(text("WITH aminoacids AS (\
                                        SELECT \
                                            psfg.protein_source_or_food_product,\
                                            pst.source_type,\
                                            pst.protein_source,\
                                            CASE \
                                                WHEN adimin.\"Alanine (Ala/A) (g/100 g Protein)\" IS NULL THEN atv.\"Alanine (Ala/A) (g/100 g Protein)\"\
                                                ELSE adimin.\"Alanine (Ala/A) (g/100 g Protein)\"\
                                            END \"Alanine (Ala/A) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Arginine  (Arg/R) (g/100 g Protein)\" IS NULL THEN atv.\"Arginine  (Arg/R) (g/100 g Protein)\"\
                                                ELSE adimin.\"Arginine  (Arg/R) (g/100 g Protein)\"\
                                            END \"Arginine  (Arg/R) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Aspartic Acid (Asp/D) (g/100 g Protein)\" IS NULL THEN atv.\"Aspartic Acid (Asp/D) (g/100 g Protein)\"\
                                                ELSE adimin.\"Aspartic Acid (Asp/D) (g/100 g Protein)\"\
                                            END \"Aspartic Acid (Asp/D) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Glutamic acid (Glu/E) (g/100 g Protein)\" IS NULL THEN atv.\"Glutamic acid (Glu/E) (g/100 g Protein)\"\
                                                ELSE adimin.\"Glutamic acid (Glu/E) (g/100 g Protein)\"\
                                            END \"Glutamic acid (Glu/E) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Glycine (Gly/G) (g/100 g Protein)\" IS NULL THEN atv.\"Glycine (Gly/G) (g/100 g Protein)\"\
                                                ELSE adimin.\"Glycine (Gly/G) (g/100 g Protein)\"\
                                            END \"Glycine (Gly/G) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Histidine  (His/H) (g/100 g Protein)\" IS NULL THEN atv.\"Histidine  (His/H) (g/100 g Protein)\"\
                                                ELSE adimin.\"Histidine  (His/H) (g/100 g Protein)\"\
                                            END \"Histidine  (His/H) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Isoleucine (Ile/I) (g/100 g Protein)\" IS NULL THEN atv.\"Isoleucine (Ile/I) (g/100 g Protein)\"\
                                                ELSE adimin.\"Isoleucine (Ile/I) (g/100 g Protein)\"\
                                            END \"Isoleucine (Ile/I) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Leucine (Leu/L) (g/100 g Protein)\" IS NULL THEN atv.\"Leucine (Leu/L) (g/100 g Protein)\"\
                                                ELSE adimin.\"Leucine (Leu/L) (g/100 g Protein)\"\
                                            END \"Leucine (Leu/L) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Lysine (Lys/K) (g/100 g Protein)\" IS NULL THEN atv.\"Lysine (Lys/K) (g/100 g Protein)\"\
                                                ELSE adimin.\"Lysine (Lys/K) (g/100 g Protein)\"\
                                            END \"Lysine (Lys/K) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Methionine (Met/M) (g/100 g Protein)\" IS NULL THEN atv.\"Methionine (Met/M) (g/100 g Protein)\"\
                                                ELSE adimin.\"Methionine (Met/M) (g/100 g Protein)\"\
                                            END \"Methionine (Met/M) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Phenylalanine (Phe/F) (g/100 g Protein)\" IS NULL THEN atv.\"Phenylalanine (Phe/F) (g/100 g Protein)\"\
                                                ELSE adimin.\"Phenylalanine (Phe/F) (g/100 g Protein)\"\
                                            END \"Phenylalanine (Phe/F) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Proline (Pro/P) (g/100 g Protein)\" IS NULL THEN atv.\"Proline (Pro/P) (g/100 g Protein)\"\
                                                ELSE adimin.\"Proline (Pro/P) (g/100 g Protein)\"\
                                            END \"Proline (Pro/P) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Serine (Ser/S) (g/100 g Protein)\" IS NULL THEN atv.\"Serine (Ser/S) (g/100 g Protein)\"\
                                                ELSE adimin.\"Serine (Ser/S) (g/100 g Protein)\"\
                                            END \"Serine (Ser/S) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Threonine (Thr/T) (g/100 g Protein)\" IS NULL THEN atv.\"Threonine (Thr/T) (g/100 g Protein)\"\
                                                ELSE adimin.\"Threonine (Thr/T) (g/100 g Protein)\"\
                                            END \"Threonine (Thr/T) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Valine (Val/V) (g/100 g Protein)\" IS NULL THEN atv.\"Valine (Val/V) (g/100 g Protein)\"\
                                                ELSE adimin.\"Valine (Val/V) (g/100 g Protein)\"\
                                            END \"Valine (Val/V) (g/100 g Protein)\"\
                                        FROM \
                                            protein_source_format_gm psfg\
                                        INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                        LEFT JOIN public.aminoacid_data_imputation_model_input adimin ON psfg.protein_source_or_food_product = adimin.protein_source_or_food_product \
                                        LEFT JOIN public.aminoacids_theoretical_values atv ON pst.protein_source = atv.protein_source\
                                        )\
                                        SELECT\
                                            psfg.id,\
                                            psfg.category,\
                                            ndim.*,\
                                            aa.*\
                                        FROM\
                                            protein_source_format_gm psfg\
                                        INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                        INNER JOIN public.nutrients_data_imputation_model_input ndim ON ndim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        INNER JOIN aminoacids aa ON aa.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        WHERE\
                                            pst.source_type = 'Traditional';") , con=connection)
        
        test_nutrients_aminoacids_environmental_data_model = pd.read_sql_query(text("WITH aminoacids AS (\
                                        SELECT \
                                            psfg.protein_source_or_food_product,\
                                            pst.source_type,\
                                            pst.protein_source,\
                                            CASE \
                                                WHEN adimin.\"Alanine (Ala/A) (g/100 g Protein)\" IS NULL THEN atv.\"Alanine (Ala/A) (g/100 g Protein)\"\
                                                ELSE adimin.\"Alanine (Ala/A) (g/100 g Protein)\"\
                                            END \"Alanine (Ala/A) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Arginine  (Arg/R) (g/100 g Protein)\" IS NULL THEN atv.\"Arginine  (Arg/R) (g/100 g Protein)\"\
                                                ELSE adimin.\"Arginine  (Arg/R) (g/100 g Protein)\"\
                                            END \"Arginine  (Arg/R) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Aspartic Acid (Asp/D) (g/100 g Protein)\" IS NULL THEN atv.\"Aspartic Acid (Asp/D) (g/100 g Protein)\"\
                                                ELSE adimin.\"Aspartic Acid (Asp/D) (g/100 g Protein)\"\
                                            END \"Aspartic Acid (Asp/D) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Glutamic acid (Glu/E) (g/100 g Protein)\" IS NULL THEN atv.\"Glutamic acid (Glu/E) (g/100 g Protein)\"\
                                                ELSE adimin.\"Glutamic acid (Glu/E) (g/100 g Protein)\"\
                                            END \"Glutamic acid (Glu/E) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Glycine (Gly/G) (g/100 g Protein)\" IS NULL THEN atv.\"Glycine (Gly/G) (g/100 g Protein)\"\
                                                ELSE adimin.\"Glycine (Gly/G) (g/100 g Protein)\"\
                                            END \"Glycine (Gly/G) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Histidine  (His/H) (g/100 g Protein)\" IS NULL THEN atv.\"Histidine  (His/H) (g/100 g Protein)\"\
                                                ELSE adimin.\"Histidine  (His/H) (g/100 g Protein)\"\
                                            END \"Histidine  (His/H) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Isoleucine (Ile/I) (g/100 g Protein)\" IS NULL THEN atv.\"Isoleucine (Ile/I) (g/100 g Protein)\"\
                                                ELSE adimin.\"Isoleucine (Ile/I) (g/100 g Protein)\"\
                                            END \"Isoleucine (Ile/I) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Leucine (Leu/L) (g/100 g Protein)\" IS NULL THEN atv.\"Leucine (Leu/L) (g/100 g Protein)\"\
                                                ELSE adimin.\"Leucine (Leu/L) (g/100 g Protein)\"\
                                            END \"Leucine (Leu/L) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Lysine (Lys/K) (g/100 g Protein)\" IS NULL THEN atv.\"Lysine (Lys/K) (g/100 g Protein)\"\
                                                ELSE adimin.\"Lysine (Lys/K) (g/100 g Protein)\"\
                                            END \"Lysine (Lys/K) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Methionine (Met/M) (g/100 g Protein)\" IS NULL THEN atv.\"Methionine (Met/M) (g/100 g Protein)\"\
                                                ELSE adimin.\"Methionine (Met/M) (g/100 g Protein)\"\
                                            END \"Methionine (Met/M) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Phenylalanine (Phe/F) (g/100 g Protein)\" IS NULL THEN atv.\"Phenylalanine (Phe/F) (g/100 g Protein)\"\
                                                ELSE adimin.\"Phenylalanine (Phe/F) (g/100 g Protein)\"\
                                            END \"Phenylalanine (Phe/F) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Proline (Pro/P) (g/100 g Protein)\" IS NULL THEN atv.\"Proline (Pro/P) (g/100 g Protein)\"\
                                                ELSE adimin.\"Proline (Pro/P) (g/100 g Protein)\"\
                                            END \"Proline (Pro/P) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Serine (Ser/S) (g/100 g Protein)\" IS NULL THEN atv.\"Serine (Ser/S) (g/100 g Protein)\"\
                                                ELSE adimin.\"Serine (Ser/S) (g/100 g Protein)\"\
                                            END \"Serine (Ser/S) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Threonine (Thr/T) (g/100 g Protein)\" IS NULL THEN atv.\"Threonine (Thr/T) (g/100 g Protein)\"\
                                                ELSE adimin.\"Threonine (Thr/T) (g/100 g Protein)\"\
                                            END \"Threonine (Thr/T) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Valine (Val/V) (g/100 g Protein)\" IS NULL THEN atv.\"Valine (Val/V) (g/100 g Protein)\"\
                                                ELSE adimin.\"Valine (Val/V) (g/100 g Protein)\"\
                                            END \"Valine (Val/V) (g/100 g Protein)\"\
                                        FROM \
                                            protein_source_format_gm psfg\
                                        INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                        LEFT JOIN public.aminoacid_data_imputation_model_input adimin ON psfg.protein_source_or_food_product = adimin.protein_source_or_food_product \
                                        LEFT JOIN public.aminoacids_theoretical_values atv ON pst.protein_source = atv.protein_source\
                                        ),\
                                        environmental_impact as (\
                                        select\
                                            psfg.protein_source_or_food_product,\
                                            pst.source_type,\
                                            pst.protein_source,\
                                            CASE \
                                                WHEN edimin.\"Climate change (kg CO2 eq)\" IS NULL THEN etv.\"Climate change (kg CO2 eq)\"\
                                                ELSE edimin.\"Climate change (kg CO2 eq)\"\
                                            END \"Climate change (kg CO2 eq)\"\
                                        FROM \
                                            protein_source_format_gm psfg\
                                        INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                        LEFT JOIN public.environmental_data_imputation_model_input edimin ON psfg.protein_source_or_food_product = edimin.protein_source_or_food_product \
                                        LEFT JOIN public.environmental_theoretical_values etv ON pst.protein_source = etv.protein_source\
                                        )\
                                        SELECT\
                                            psfg.id,\
                                            psfg.category,\
                                            ndim.*,\
                                            aa.*,\
                                            ei.*\
                                        FROM\
                                            protein_source_format_gm psfg\
                                        INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                        INNER JOIN public.nutrients_data_imputation_model_input ndim ON ndim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        INNER JOIN aminoacids aa ON aa.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        INNER JOIN environmental_impact ei ON ei.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        WHERE\
                                            pst.source_type = 'Traditional';") , con=connection)


        trans.commit()

    df_test_data_model = pd.DataFrame(test_nutrients_data_model)
    #df_test_data_model = pd.DataFrame(test_nutrients_aminoacids_data_model)
    #df_test_data_model = pd.DataFrame(test_nutrients_aminoacids_environmental_data_model)

    df_sample_test_data_model = df_test_data_model.sample(n=1000, random_state=42).reset_index()

    selected_cols = df_test_data_model.select_dtypes(include=["float"]).columns


    # 2 - Select nutrient, amino acid, environmental impact data already treated for the model. Exclude traditional protein sources and the same category as the input protein
    with giantleaps_engine.connect() as connection:
        trans = connection.begin()

        result_nutrients_data_model = pd.read_sql_query(text("SELECT\
                                                                psfg.id,\
                                                                ndim.*\
                                                            FROM\
                                                                public.nutrients_data_imputation_model_input ndim\
                                                            INNER JOIN protein_source_format_gm psfg ON ndim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                                            INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                                            WHERE\
                                                                pst.source_type != 'Traditional';") , con=connection)
        result_nutrients_aminoacids_data_model = pd.read_sql_query(text("WITH aminoacids AS (\
                                        SELECT \
                                            psfg.protein_source_or_food_product,\
                                            pst.source_type,\
                                            pst.protein_source,\
                                            CASE \
                                                WHEN adimin.\"Alanine (Ala/A) (g/100 g Protein)\" IS NULL THEN atv.\"Alanine (Ala/A) (g/100 g Protein)\"\
                                                ELSE adimin.\"Alanine (Ala/A) (g/100 g Protein)\"\
                                            END \"Alanine (Ala/A) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Arginine  (Arg/R) (g/100 g Protein)\" IS NULL THEN atv.\"Arginine  (Arg/R) (g/100 g Protein)\"\
                                                ELSE adimin.\"Arginine  (Arg/R) (g/100 g Protein)\"\
                                            END \"Arginine  (Arg/R) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Aspartic Acid (Asp/D) (g/100 g Protein)\" IS NULL THEN atv.\"Aspartic Acid (Asp/D) (g/100 g Protein)\"\
                                                ELSE adimin.\"Aspartic Acid (Asp/D) (g/100 g Protein)\"\
                                            END \"Aspartic Acid (Asp/D) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Glutamic acid (Glu/E) (g/100 g Protein)\" IS NULL THEN atv.\"Glutamic acid (Glu/E) (g/100 g Protein)\"\
                                                ELSE adimin.\"Glutamic acid (Glu/E) (g/100 g Protein)\"\
                                            END \"Glutamic acid (Glu/E) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Glycine (Gly/G) (g/100 g Protein)\" IS NULL THEN atv.\"Glycine (Gly/G) (g/100 g Protein)\"\
                                                ELSE adimin.\"Glycine (Gly/G) (g/100 g Protein)\"\
                                            END \"Glycine (Gly/G) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Histidine  (His/H) (g/100 g Protein)\" IS NULL THEN atv.\"Histidine  (His/H) (g/100 g Protein)\"\
                                                ELSE adimin.\"Histidine  (His/H) (g/100 g Protein)\"\
                                            END \"Histidine  (His/H) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Isoleucine (Ile/I) (g/100 g Protein)\" IS NULL THEN atv.\"Isoleucine (Ile/I) (g/100 g Protein)\"\
                                                ELSE adimin.\"Isoleucine (Ile/I) (g/100 g Protein)\"\
                                            END \"Isoleucine (Ile/I) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Leucine (Leu/L) (g/100 g Protein)\" IS NULL THEN atv.\"Leucine (Leu/L) (g/100 g Protein)\"\
                                                ELSE adimin.\"Leucine (Leu/L) (g/100 g Protein)\"\
                                            END \"Leucine (Leu/L) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Lysine (Lys/K) (g/100 g Protein)\" IS NULL THEN atv.\"Lysine (Lys/K) (g/100 g Protein)\"\
                                                ELSE adimin.\"Lysine (Lys/K) (g/100 g Protein)\"\
                                            END \"Lysine (Lys/K) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Methionine (Met/M) (g/100 g Protein)\" IS NULL THEN atv.\"Methionine (Met/M) (g/100 g Protein)\"\
                                                ELSE adimin.\"Methionine (Met/M) (g/100 g Protein)\"\
                                            END \"Methionine (Met/M) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Phenylalanine (Phe/F) (g/100 g Protein)\" IS NULL THEN atv.\"Phenylalanine (Phe/F) (g/100 g Protein)\"\
                                                ELSE adimin.\"Phenylalanine (Phe/F) (g/100 g Protein)\"\
                                            END \"Phenylalanine (Phe/F) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Proline (Pro/P) (g/100 g Protein)\" IS NULL THEN atv.\"Proline (Pro/P) (g/100 g Protein)\"\
                                                ELSE adimin.\"Proline (Pro/P) (g/100 g Protein)\"\
                                            END \"Proline (Pro/P) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Serine (Ser/S) (g/100 g Protein)\" IS NULL THEN atv.\"Serine (Ser/S) (g/100 g Protein)\"\
                                                ELSE adimin.\"Serine (Ser/S) (g/100 g Protein)\"\
                                            END \"Serine (Ser/S) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Threonine (Thr/T) (g/100 g Protein)\" IS NULL THEN atv.\"Threonine (Thr/T) (g/100 g Protein)\"\
                                                ELSE adimin.\"Threonine (Thr/T) (g/100 g Protein)\"\
                                            END \"Threonine (Thr/T) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Valine (Val/V) (g/100 g Protein)\" IS NULL THEN atv.\"Valine (Val/V) (g/100 g Protein)\"\
                                                ELSE adimin.\"Valine (Val/V) (g/100 g Protein)\"\
                                            END \"Valine (Val/V) (g/100 g Protein)\"\
                                        FROM \
                                            protein_source_format_gm psfg\
                                        INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                        LEFT JOIN public.aminoacid_data_imputation_model_input adimin ON psfg.protein_source_or_food_product = adimin.protein_source_or_food_product \
                                        LEFT JOIN public.aminoacids_theoretical_values atv ON pst.protein_source = atv.protein_source\
                                        )\
                                        SELECT\
                                            psfg.id,\
                                            psfg.category,\
                                            ndim.*,\
                                            aa.*\
                                        FROM\
                                            protein_source_format_gm psfg\
                                        INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                        INNER JOIN public.nutrients_data_imputation_model_input ndim ON ndim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        INNER JOIN aminoacids aa ON aa.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        WHERE\
                                            pst.source_type != 'Traditional';"), con=connection)
    

        result_nutrients_aminoacids_environmental_data_model = pd.read_sql_query(text("WITH aminoacids AS (\
                                        SELECT \
                                            psfg.protein_source_or_food_product,\
                                            pst.source_type,\
                                            pst.protein_source,\
                                            CASE \
                                                WHEN adimin.\"Alanine (Ala/A) (g/100 g Protein)\" IS NULL THEN atv.\"Alanine (Ala/A) (g/100 g Protein)\"\
                                                ELSE adimin.\"Alanine (Ala/A) (g/100 g Protein)\"\
                                            END \"Alanine (Ala/A) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Arginine  (Arg/R) (g/100 g Protein)\" IS NULL THEN atv.\"Arginine  (Arg/R) (g/100 g Protein)\"\
                                                ELSE adimin.\"Arginine  (Arg/R) (g/100 g Protein)\"\
                                            END \"Arginine  (Arg/R) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Aspartic Acid (Asp/D) (g/100 g Protein)\" IS NULL THEN atv.\"Aspartic Acid (Asp/D) (g/100 g Protein)\"\
                                                ELSE adimin.\"Aspartic Acid (Asp/D) (g/100 g Protein)\"\
                                            END \"Aspartic Acid (Asp/D) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Glutamic acid (Glu/E) (g/100 g Protein)\" IS NULL THEN atv.\"Glutamic acid (Glu/E) (g/100 g Protein)\"\
                                                ELSE adimin.\"Glutamic acid (Glu/E) (g/100 g Protein)\"\
                                            END \"Glutamic acid (Glu/E) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Glycine (Gly/G) (g/100 g Protein)\" IS NULL THEN atv.\"Glycine (Gly/G) (g/100 g Protein)\"\
                                                ELSE adimin.\"Glycine (Gly/G) (g/100 g Protein)\"\
                                            END \"Glycine (Gly/G) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Histidine  (His/H) (g/100 g Protein)\" IS NULL THEN atv.\"Histidine  (His/H) (g/100 g Protein)\"\
                                                ELSE adimin.\"Histidine  (His/H) (g/100 g Protein)\"\
                                            END \"Histidine  (His/H) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Isoleucine (Ile/I) (g/100 g Protein)\" IS NULL THEN atv.\"Isoleucine (Ile/I) (g/100 g Protein)\"\
                                                ELSE adimin.\"Isoleucine (Ile/I) (g/100 g Protein)\"\
                                            END \"Isoleucine (Ile/I) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Leucine (Leu/L) (g/100 g Protein)\" IS NULL THEN atv.\"Leucine (Leu/L) (g/100 g Protein)\"\
                                                ELSE adimin.\"Leucine (Leu/L) (g/100 g Protein)\"\
                                            END \"Leucine (Leu/L) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Lysine (Lys/K) (g/100 g Protein)\" IS NULL THEN atv.\"Lysine (Lys/K) (g/100 g Protein)\"\
                                                ELSE adimin.\"Lysine (Lys/K) (g/100 g Protein)\"\
                                            END \"Lysine (Lys/K) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Methionine (Met/M) (g/100 g Protein)\" IS NULL THEN atv.\"Methionine (Met/M) (g/100 g Protein)\"\
                                                ELSE adimin.\"Methionine (Met/M) (g/100 g Protein)\"\
                                            END \"Methionine (Met/M) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Phenylalanine (Phe/F) (g/100 g Protein)\" IS NULL THEN atv.\"Phenylalanine (Phe/F) (g/100 g Protein)\"\
                                                ELSE adimin.\"Phenylalanine (Phe/F) (g/100 g Protein)\"\
                                            END \"Phenylalanine (Phe/F) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Proline (Pro/P) (g/100 g Protein)\" IS NULL THEN atv.\"Proline (Pro/P) (g/100 g Protein)\"\
                                                ELSE adimin.\"Proline (Pro/P) (g/100 g Protein)\"\
                                            END \"Proline (Pro/P) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Serine (Ser/S) (g/100 g Protein)\" IS NULL THEN atv.\"Serine (Ser/S) (g/100 g Protein)\"\
                                                ELSE adimin.\"Serine (Ser/S) (g/100 g Protein)\"\
                                            END \"Serine (Ser/S) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Threonine (Thr/T) (g/100 g Protein)\" IS NULL THEN atv.\"Threonine (Thr/T) (g/100 g Protein)\"\
                                                ELSE adimin.\"Threonine (Thr/T) (g/100 g Protein)\"\
                                            END \"Threonine (Thr/T) (g/100 g Protein)\",\
                                            CASE \
                                                WHEN adimin.\"Valine (Val/V) (g/100 g Protein)\" IS NULL THEN atv.\"Valine (Val/V) (g/100 g Protein)\"\
                                                ELSE adimin.\"Valine (Val/V) (g/100 g Protein)\"\
                                            END \"Valine (Val/V) (g/100 g Protein)\"\
                                        FROM \
                                            protein_source_format_gm psfg\
                                        INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                        LEFT JOIN public.aminoacid_data_imputation_model_input adimin ON psfg.protein_source_or_food_product = adimin.protein_source_or_food_product \
                                        LEFT JOIN public.aminoacids_theoretical_values atv ON pst.protein_source = atv.protein_source\
                                        ), environmental_impact as (\
                                            select\
                                                psfg.protein_source_or_food_product,\
                                                pst.source_type,\
                                                pst.protein_source,\
                                                CASE \
                                                    WHEN edimin.\"Climate change (kg CO2 eq)\" IS NULL THEN etv.\"Climate change (kg CO2 eq)\"\
                                                    ELSE edimin.\"Climate change (kg CO2 eq)\"\
                                                END \"Climate change (kg CO2 eq)\"\
                                            FROM \
                                                protein_source_format_gm psfg\
                                            INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                            LEFT JOIN public.environmental_data_imputation_model_input edimin ON psfg.protein_source_or_food_product = edimin.protein_source_or_food_product \
                                            LEFT JOIN public.environmental_theoretical_values etv ON pst.protein_source = etv.protein_source\
                                        )\
                                        SELECT\
                                            psfg.id,\
                                            psfg.category,\
                                            ndim.*,\
                                            aa.*,\
                                            ei.*\
                                        FROM\
                                            protein_source_format_gm psfg\
                                        INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                        INNER JOIN public.nutrients_data_imputation_model_input ndim ON ndim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        INNER JOIN aminoacids aa ON aa.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        INNER JOIN environmental_impact ei ON ei.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        WHERE\
                                            pst.source_type != 'Traditional';"), con=connection)
  
  
        trans.commit()

    df_protein_data_model = pd.DataFrame(result_nutrients_data_model)
    #df_protein_data_model = pd.DataFrame(result_nutrients_aminoacids_data_model)
    #df_protein_data_model = pd.DataFrame(result_nutrients_aminoacids_environmental_data_model)

    # 3 - Prepare data for k-NN model
         # drop duplicates based on nutrient and amino acid values, not on id, category, etc. to avoid having repeated rows with the same nutritional profile
         # drop rows with missing values in nutrient and amino acid columns, as k-NN cannot handle them. We can impute them later if we want to include those products in the model
    idx_unique_proteins = df_protein_data_model.select_dtypes(include='float').drop_duplicates().index
    X_model = df_protein_data_model.loc[idx_unique_proteins,:].select_dtypes(include='float').values
    X_model = X_model.astype('float32')
    y_model = df_protein_data_model.loc[idx_unique_proteins,'id'].values

    # 3.1 - Prepare catalog data for metrics
    df_protein_data_model = df_protein_data_model.set_index(df_protein_data_model['id'])
    catalog = df_protein_data_model.select_dtypes(include='float').drop_duplicates().to_dict(orient='index')

    for name, pipe, param_grid in make_model_candidates():
        pipe.fit(X_model, y_model)
        print(f"Model {name} trained.")
        df_sample_test_data_model[f'{name}_recommendation'] = None
        df_sample_test_data_model[f'{name}_distances'] = None
        list_diversity = []
        list_distances = []
        for index, row in df_sample_test_data_model.iterrows():
            #print(row.id, row.category)
            X_predict = row[selected_cols].values
            #print(X_predict)
            prediction = pipe.predict(X_predict.reshape(1, -1))
            #print(f"Prediction for {row.id}: {prediction}")
            recommendations, distances = prediction[0]
            col_rec_idx = df_sample_test_data_model.columns.get_loc(f'{name}_recommendation')
            col_dist_idx = df_sample_test_data_model.columns.get_loc(f'{name}_distances')
            df_sample_test_data_model.iat[index, col_rec_idx] = recommendations # Assuming prediction is a list of lists
            df_sample_test_data_model.iat[index, col_dist_idx] = distances # Assuming prediction is a list of lists

        list_diversity = df_sample_test_data_model[f'{name}_recommendation'].apply(compute_individual_diversity, args=[catalog])
        diversity = np.mean(list_diversity)
        print(f"Individual diversity for model {name}: {diversity}")

        list_distances = df_sample_test_data_model[f'{name}_distances'].apply(lambda x: np.mean(x))
        mean_distance = np.mean(list_distances)
        print(f"Mean distance for model {name}: {mean_distance}")