
def load_parameters():

    db_conn_file = 'ai_giantleaps/others/db_credentials.txt'
    db_name = REMOVED
    db_type = 'PostgreSQL'

    pca_dir = "others"
    pca_file = "others/pca_giant.xlsx"
    mean_file = "others/pca_mean.xlsx"
    scale_file = "others/pca_scale.xlsx"

    longListAminoAcids = 'others/longList_aminoacids.csv'
    longListNutrients = 'others/longList_nutrients.csv'
    longListAntinutrients = 'others/longList_antinutritional_factors.csv'
    shortListAminoAcids = 'others/shortList_aminoacids.csv'
    shortListNutrients = 'others/shortList_nutrients.csv'
    shortListAntinutrients = 'others/shortList_antinutritional_factors.csv'

    unitsNutrientsDb = 'others/units_nutrients_db.csv'

    parameters = locals().copy()
    return parameters