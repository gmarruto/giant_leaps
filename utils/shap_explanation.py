import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# ============================================================
# SHAP explanation for pipe_pca75_knn_custom
# ============================================================

def compute_shap_for_top1_recommendation(
    pipe_pca_knn_custom,
    X_model,
    y_model,
    X_predict,
    y_predict,
    listRecommendations,
    feature_names,
    background_size=100,
    nsamples=200,
    output_prefix="protein_screening_shap"
):
    """
    Explain why the top-1 protein recommendation is recommended
    for the input protein, using SHAP on a similarity score.

    Parameters
    ----------
    pipe_pca_knn_custom : fitted sklearn Pipeline
        Pipeline with PCA + custom KNN
    X_model : np.ndarray
        Training matrix used to fit the recommendation model
    y_model : np.ndarray
        IDs of training proteins
    X_predict : np.ndarray
        Matrix to predict (normally one protein, but can be many)
    y_predict : np.ndarray
        IDs of proteins to explain
    listRecommendations : output of pipe_pca_knn_custom.predict(X_predict)
        Expected structure like:
        [
            [[rec_id_1, rec_id_2, ..., rec_id_k]],
            ...
        ]
    feature_names : list[str]
        Original numeric feature names used in X_model / X_predict
    background_size : int
        Number of background samples for Kernel SHAP
    nsamples : int
        Number of samples for Kernel SHAP
    output_prefix : str
        Prefix for saved files

    Returns
    -------
    dict
        Dictionary with SHAP results and metadata
    """

    # -----------------------------
    # 1) Extract fitted PCA
    # -----------------------------
    pca_step = pipe_pca_knn_custom.named_steps["pca"]

    # -----------------------------
    # 2) Transform train and predict data to PCA space
    # -----------------------------
    X_model_pca = pca_step.transform(X_model)
    X_predict_pca = pca_step.transform(X_predict)

    # -----------------------------
    # 3) Build DataFrames for safer handling
    # -----------------------------
    X_model_df = pd.DataFrame(X_model, columns=feature_names)
    X_predict_df = pd.DataFrame(X_predict, columns=feature_names)

    # -----------------------------
    # 4) Background sample for SHAP
    # -----------------------------
    if len(X_model_df) > background_size:
        background_df = X_model_df.sample(background_size, random_state=42)
    else:
        background_df = X_model_df.copy()

    # -----------------------------
    # 5) Helper: score function to explain
    #    score(x) = - euclidean distance to top-1 recommended protein in PCA space
    # -----------------------------
    def build_top1_score_function(target_pca_vector):
        def score_fn(X_raw):
            X_raw = np.asarray(X_raw, dtype=np.float32)

            # Apply only PCA because data is already normalised in your pipeline
            X_pca = pca_step.transform(X_raw)

            # Negative euclidean distance -> higher score = more similar
            distances = cdist(X_pca, target_pca_vector.reshape(1, -1), metric="euclidean").ravel()
            scores = -distances
            return scores
        return score_fn

    # -----------------------------
    # 6) Explain each predicted protein input
    # -----------------------------
    all_results = []

    for i in range(len(X_predict_df)):
        input_id = y_predict[i]
        input_row_df = X_predict_df.iloc[[i]]

        # Expected structure from your code:
        # for recomm in listRecommendations:
        #     for protein_id in recomm[0]:
        #
        # so here the top-1 should be listRecommendations[i][0][0]
        try:
            top1_rec_id = listRecommendations[i][0][0]
        except Exception:
            raise ValueError(
                f"No he podido leer la recomendación top-1 para el índice {i}. "
                f"Estructura recibida: {listRecommendations[i]}"
            )

        # Locate recommended protein in training set
        rec_matches = np.where(y_model == top1_rec_id)[0]
        if len(rec_matches) == 0:
            raise ValueError(
                f"La proteína recomendada top-1 con id={top1_rec_id} no está en y_model."
            )

        rec_idx = rec_matches[0]
        rec_vector_pca = X_model_pca[rec_idx]

        # Build score function for this specific top-1 recommendation
        score_fn = build_top1_score_function(rec_vector_pca)

        # Kernel SHAP on original feature space
        explainer = shap.KernelExplainer(score_fn, background_df.values)
        shap_values = explainer.shap_values(input_row_df.values, nsamples=nsamples)

        # Convert output to 1D
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 2:
            shap_vector = shap_values[0]
        elif shap_values.ndim == 3:
            shap_vector = shap_values[0, 0, :]
        else:
            shap_vector = shap_values.ravel()

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = np.asarray(expected_value).ravel()[0]

        # Create results DataFrame
        result_df = pd.DataFrame({
            "feature": feature_names,
            "feature_value": input_row_df.iloc[0].values,
            "shap_value": shap_vector,
            "abs_shap_value": np.abs(shap_vector)
        }).sort_values("abs_shap_value", ascending=False)

        # Save CSV
        csv_name = f"{output_prefix}_input_{input_id}_top1rec_{top1_rec_id}.csv"
        result_df.to_csv(csv_name, index=False)

        # Save waterfall plot
        explanation = shap.Explanation(
            values=shap_vector,
            base_values=expected_value,
            data=input_row_df.iloc[0].values,
            feature_names=feature_names
        )

        plt.figure()
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(
            f"{output_prefix}_input_{input_id}_top1rec_{top1_rec_id}_waterfall.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

        # Save bar plot
        plt.figure()
        shap.plots.bar(explanation, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(
            f"{output_prefix}_input_{input_id}_top1rec_{top1_rec_id}_bar.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

        all_results.append({
            "input_id": input_id,
            "top1_rec_id": top1_rec_id,
            "expected_value": expected_value,
            "shap_table": result_df
        })

    return {
        "results": all_results,
        "background_rows": len(background_df),
        "nsamples": nsamples
    }