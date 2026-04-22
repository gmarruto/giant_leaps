import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocessData(longListNutrients_file, unitsNutrients_file):

    df_longListNutrients = pd.read_csv(longListNutrients_file, header = 0, names = ["protein_source_or_food_product","data_source_name","id","nutrient","nutrient_type","amount","unit"], sep=';')
    df_longList_protein_source_type = pd.DataFrame(data={"protein_source" : ['Soya beans raw','Wheat whole grain raw','Peas, green, raw','Chickpeas raw','Potato, raw','Spirulina raw','Beef, ground, raw','Pork meat raw',
                                                            'Chicken, meat, raw','Egg, whole, raw','Cow\'s milk certified raw milk', 'Fish side streams', 'Krill', 'Brewer\'s spent grain'], 
                                        "protein_type" : ['Plant-based', 'Plant-based', 'Plant-based', 'Plant-based', 'Plant-based','Ocean-based', 'Animal-based', 'Animal-based', 'Animal-based', 'Animal-based', 'Animal-based',
                                                            'Ocean-based', 'Ocean-based', 'Plant-based']})
    df_units_nutrients = pd.read_csv(unitsNutrients_file, header=0, names = ["nutrient", "unit", "data_source_name", "standardized_unit", "conversion_factor"], sep=';')
    df_corrections = pd.DataFrame( data = {"incorrect":["MG", "G", "mcg_DFE", "mcg_RAE", "KCAL", "G/100G", "KCAL/100G", "Kcal/100g", "kcal/100 g", "KJ/100g", "µg/100g", "MG/100G", "MG/100g", "Mg/100g", "mg/100 g", "g/port", "BE", "ratio", "IU", "RE (µg/100g)", "RE (µg/100g)", "alfa-TE", "NE", "%", "No Unit", "kJ", "RE/µg", "µg", "NE/mg"],
                                            "correct": ["mg", "g", "mcg", "mcg", "kcal", "g/100g", "kcal/100g", "kcal/100g","kcal/100g", "kJ/100g", "µg/100g", "mg/100g", "mg/100g", "mg/100g", "mg/100g", "g/port", "BE", "ratio", "IU", "RE (µg/100g)", "RE (µg/100g)", "alfa-TE", "NE", "%", "No Unit", "kJ", "RE/µg", "µg", "NE/mg"]}
    )


    df_longListNutrients = df_longListNutrients.dropna()
    df_longListNutrients = pd.merge(df_longListNutrients, df_longList_protein_source_type, left_on="protein_source_or_food_product", right_on="protein_source")
    # Preprocess amount column
    df_longListNutrients.amount = df_longListNutrients.amount.apply(lambda x: round(float(x.replace(',','.')), 2))
    # Preprocess unit column
    for idx, row in df_corrections.iterrows():
        df_longListNutrients.loc[idx,].unit = df_longListNutrients.unit.replace(df_corrections.loc[idx,].incorrect, df_corrections.loc[idx,].correct)
        df_units_nutrients.unit = df_units_nutrients.unit.replace(df_corrections.loc[idx,].incorrect, df_corrections.loc[idx,].correct)

    # Standardize nutrient unit measures
    df_units_nutrients = df_units_nutrients.drop_duplicates()
    df_units_nutrients.conversion_factor = df_units_nutrients.conversion_factor.apply(lambda x: float(x))

    df_longListNutrients_units = pd.merge(df_longListNutrients, df_units_nutrients, left_on = ["nutrient", "unit", "data_source_name"], right_on = ["nutrient", "unit", "data_source_name"])
    df_longListNutrients_units['amount_stand'] = df_longListNutrients_units.apply(lambda x: x.amount*x.conversion_factor, axis=1)

    df_longListNutrients_units['nutrient_unit'] = df_longListNutrients_units.nutrient + " - " + df_longListNutrients_units.standardized_unit

    # Pre-process "nutrient" column
    df_longListNutrients_order = df_longListNutrients_units.sort_values(by="nutrient")

    # Nutrient informed variable/tag
    df_longListNutrients_order['nutrient_informed'] = df_longListNutrients_order.apply(lambda x: None if np.isnan(x.amount) else x.nutrient, axis=1)

    df_longListNutrients_tbl = pd.pivot_table(data=df_longListNutrients_order,
              values='amount_stand',
              aggfunc='mean',
              columns='nutrient_unit',
              index=['protein_source_or_food_product', 'data_source_name', 'protein_type']
              ).reset_index()
    
    return df_longListNutrients_tbl


def PCA_analysis(df, pca_dir, var_colour = None, var_shape = None, plot_title = None, file_dir = None):

    df_numerics = df.select_dtypes(include='number')

    # Filter dataframe by numeric columns and columns with no NA
    # and variance different to 0
    df_PCA = df_numerics.dropna(axis=1)
    df_PCA = df_PCA[df_PCA.columns[df_PCA.var() != 0]]

    # Scale data before applying PCA
    scaling=StandardScaler()

    # Use fit and transform method, center and scale
    scaledData=scaling.fit_transform(df_PCA)

    # Calculate mean and scale
    pd.DataFrame(data=scaling.mean_.reshape(1,len(scaling.mean_)), columns=df_PCA.columns).to_excel(pca_dir + '/pca_mean.xlsx', index=False)
    pd.DataFrame(data=scaling.scale_.reshape(1,len(scaling.scale_)), columns=df_PCA.columns).to_excel(pca_dir + '/pca_scale.xlsx', index=False)

    pca = PCA(n_components=2)
    tx_data = pca.fit_transform(scaledData)

    df_pca = pd.DataFrame(data=pca.components_, columns=df_PCA.columns)
    df_pca.to_excel(pca_dir + '/pca_giant.xlsx', index=False)

    return pca, df_PCA.columns, scaling, tx_data


def load_PCA(pca_dir):
    return pd.read_excel(pca_dir+'/pca_giant.xlsx'), pd.read_excel(pca_dir+'/pca_mean.xlsx'), pd.read_excel(pca_dir+'/pca_scale.xlsx')

