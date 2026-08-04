import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from fastapi import HTTPException, status


import api_parameters
import utils.data_models as data_models
import utils.protein_screening_parameters as protein_screening_parameters
import utils.pca_analysis as pca_analysis

from sqlalchemy.sql import text

import re
from collections import Counter


api_params = api_parameters.load_parameters()
protscreen_params = protein_screening_parameters.load_parameters()


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

def AiProtScreenTrainning(target_file):
    # Database processed from where to extract the info to train the algorithm
    df_NutrientsProcessed = pca_analysis.preprocessData(api_params['longListNutrients'], api_params['unitsNutrientsDb'])

    # 1 - Compute Principal Component Analysis
    df_pca, features, df_scale, txNutrientsData = pca_analysis.PCA_analysis(df_NutrientsProcessed, api_params['pca_dir'])
    # 2 - Train the algorithm with reduced dimensions values
    NN = NearestNeighbors(n_neighbors=protscreen_params['k_neighbors'])
    NN.fit(txNutrientsData)
    joblib.dump(NN, target_file)

    return NN


def AiProtScreenLoad(load_file):

    return joblib.load(load_file, mmap_mode ='r')

def _AltProtRecommendation(data: data_models.userProteinInput):
    df_pca, df_mean, df_scale  = pca_analysis.load_PCA(api_params['pca_dir'])
    loadings = df_pca.values
    df_NutrientsProcessed = pca_analysis.preprocessData(api_params['longListNutrients'], api_params['unitsNutrientsDb'])

    scaledProtein = (data.proteinDataValues - df_mean.values[0])/df_scale.values[0]
    txInputData = np.dot(scaledProtein, loadings.T)


    try:
        NN = AiProtScreenLoad(protscreen_params['ai_model_file'])
    except FileNotFoundError:
        NN = AiProtScreenTrainning(protscreen_params['ai_model_file'])
    resultNeighbor = NN.kneighbors(txInputData.reshape(1,2))

    # Output format - alternativeProtein model
    listAltProt = data_models.ListAlernativeProtein()
    for protIdx in range(len(resultNeighbor.__getitem__(0)[0])):
        altProt = data_models.alternativeProtein()
        altProtIdx = resultNeighbor.__getitem__(1)[0][protIdx]
        setattr(altProt, 'proteinName', df_NutrientsProcessed.protein_source_or_food_product[altProtIdx])
        setattr(altProt, 'proteinContent', df_NutrientsProcessed['Protein - g'][altProtIdx])
        listAltProt.alternativeProteins.append(altProt)

    return listAltProt

# Category definition
def get_category(df):

    conditions = [
        df['protein_source_or_food_product'].str.contains('concentrate|isolate|powder|extract|raw', case=False, na=False),
        df['protein_source_or_food_product'].str.contains('bread|bar|snack|meal', case=False, na=False)
    ]

    choices = ['ingredient', 'final product']

    df['category'] = np.select(conditions, choices, default='unknown')

    return df[['protein_source_or_food_product','category']]


# Tokenization and count
def tokenize(text, custom_stopwords):
# Convert to lowercase, remove punctuation, and split into words
    words = re.findall(r'\b\w+\b', str(text).lower())
    return [word for word in words if word not in custom_stopwords]


# Function to check for intersection
def contains_word(list, word_set):
    return any(word in word_set for word in list)

def moistureComputationClassification(row):
    # Compute moisture content if not available
    if pd.isna(row['Moisture (g)']):
        # Get macronutrient values. If any are missing, assume 0
        carbohydrates = row.get('Carbohydrate (g)', 0) if row.get('Carbohydrate (g)', 0) is not None else 0
        fat = row.get('Fat (g)', 0) if row.get('Fat (g)', 0) is not None else 0
        protein = row.get('Protein (g)', 0) if row.get('Protein (g)', 0) is not None else 0
        # Assuming total weight is 100g
        moisture_content = 100 - (carbohydrates + fat + protein)
        row['Moisture (g)'] = moisture_content

    # Classify based on moisture content
    moisture = row['Moisture (g)']
    if moisture < 10:
        row['Moisture Level'] = 'Very Low'
        row['Moisture Level Num'] = 0
    elif moisture < 25:
        row['Moisture Level'] = 'Low'
        row['Moisture Level Num'] = 1
    elif moisture < 50:
        row['Moisture Level'] = 'Intermediate'
        row['Moisture Level Num'] = 2
    elif moisture < 80:
        row['Moisture Level'] = 'High'
        row['Moisture Level Num'] = 3
    else:
        row['Moisture Level'] = 'Very High'
        row['Moisture Level Num'] = 4

    return row

def semanticDataAnalysis(engine):
    # Semantic analysis to extract the category of the item (product or ingredient)
    print("Starting semantic data analysis...")
    # Load food sources data
    selectProteinSources = "SELECT\
                            *\
                        FROM\
	                        public.protein_source_format_gm psfg"

    with engine.connect() as connection:
        trans = connection.begin()
        result_proteinSources = pd.read_sql_query(text(selectProteinSources), con=connection)
        trans.commit()

    df_food_sources = pd.DataFrame(result_proteinSources)
    df_food_sources = get_category(df_food_sources)

    # Protein list of words
    df_food_sources['protein_source_or_food_product'] = df_food_sources.apply(lambda x: str(x.protein_source_or_food_product) if isinstance(x.protein_source_or_food_product, float) else x.protein_source_or_food_product, axis=1)
    df_food_sources['protein_source_or_food_product_split'] = df_food_sources.apply(lambda x: ",".join(x.protein_source_or_food_product.split()), axis=1)
    df_food_sources['protein_source_or_food_product_wordlist'] = df_food_sources.protein_source_or_food_product_split.str.replace(',,', ',').str.split(',')

    # Apply tokenization to protein item columns
    all_words = df_food_sources['protein_source_or_food_product'].dropna().apply(tokenize, args=[protscreen_params['custom_stopwords']])
    # Flat the list and count
    word_freq = Counter([word for sublist in all_words for word in sublist])
    # Convert to an ordered DataFrame
    word_freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'count']).sort_values(by='count', ascending=False).reset_index(drop=True)
    # Create a set of words
    word_set = set(word_freq_df.word.values)

    # Filter DataFrame by proteins whose name is on the list
    filtrado = df_food_sources[df_food_sources['protein_source_or_food_product_wordlist'].apply(contains_word, args=[word_set])]

    filtrado.to_sql('tmp_protein_sources_category', engine, if_exists='replace', index=False)
    # UPDATE/INSERT original table with the categories
    update_protein_sources = "UPDATE public.protein_source_format_gm \
                                SET \
                                    category = public.tmp_protein_sources_category.category \
                                FROM \
                                    public.tmp_protein_sources_category \
                                WHERE public.tmp_protein_sources_category.protein_source_or_food_product = public.protein_source_format_gm.protein_source_or_food_product;"

    insert_protein_sources = "INSERT INTO public.protein_source_format_gm(protein_source_or_food_product, category) \
                                SELECT \
                                    public.tmp_protein_sources_category.protein_source_or_food_product, \
                                    public.tmp_protein_sources_category.category \
                                FROM \
                                    public.tmp_protein_sources_category \
                                WHERE public.tmp_protein_sources_category.protein_source_or_food_product NOT IN  (SELECT protein_source_or_food_product FROM public.protein_source_format_gm);"

    with engine.connect() as connection:
        trans = connection.begin()
        connection.execute(text(update_protein_sources))
        connection.execute(text(insert_protein_sources))
        trans.commit()

    print("Semantic data analysis completed. Protein sources updated with categories.")
    return

def dataImputationModelInput(engine):
    pass


def getInputProteinNutrientData(id, engine):

    selectProteinSources = "SELECT \
                                fs2.id,\
                                fs2.protein_source_or_food_product,\
                                psfg.protein_source \
                            FROM \
	                            dcf_data.food_sources fs2 \
                            LEFT JOIN protein_source_format_gm psfg ON fs2.id = psfg.id \
                            WHERE \
	                            fs2.id = '{}'".format(id)

    selectProteinRelations = "SELECT\
                                *\
                            FROM\
                                dcf_data.food_data_source_relations fdsr\
                            WHERE \
                                fdsr.food_product_id = '{}';".format(id)

    selectNutrientsComposition = "SELECT\
                                    fs2.protein_source_or_food_product,\
                                    ncf.food_data_source_relation_id,\
                                    n.id AS \"nutrient_id\",\
                                    n.nutrient,\
                                    ncf.amount AS \"amount\",\
                                    ncf.unit AS \"unit\"\
                                FROM\
                                    dcf_data.food_sources fs2\
                                INNER JOIN dcf_data.food_data_source_relations fdsr ON fdsr.food_product_id = fs2.id\
                                INNER JOIN dcf_data.nutrient_composition_facts ncf ON ncf.food_data_source_relation_id = fdsr.id \
                                INNER JOIN dcf_data.nutrients n ON n.id = ncf.nutrient_id\
                                INNER JOIN public.nutrients_units_conversion nuc ON (nuc.nutrient_name = n.nutrient AND nuc.unit = ncf.unit)\
                                WHERE nuc.conversion_factor IS NOT NULL\
                                AND fs2.id = '{}';".format(id)
    
    selectNutrients = "SELECT\
                            *\
                        FROM\
                            dcf_data.nutrients n;"

    selectNutrientsUnits = "SELECT DISTINCT\
                                    n.nutrient,\
                                    ncf.unit\
                                FROM\
                                    dcf_data.nutrient_composition_facts ncf\
                                INNER JOIN dcf_data.nutrients n ON n.id = ncf.nutrient_id\
                                INNER JOIN public.nutrients_units_conversion nuc ON (nuc.nutrient_name = n.nutrient AND nuc.unit = ncf.unit)\
                                WHERE nuc.conversion_factor IS NOT NULL"

    selectNutrientsConversion = "SELECT\
                                    *\
                                FROM\
                                    public.nutrients_units_conversion nc;"


    with engine.connect() as connection:
        trans = connection.begin()
        result_proteinSources = pd.read_sql_query(text(selectProteinSources), con=connection)
        result_proteinRelations = pd.read_sql_query(text(selectProteinRelations), con=connection)
        result_nutrientsComposition = pd.read_sql_query(text(selectNutrientsComposition), con=connection)
        result_nutrients = pd.read_sql_query(text(selectNutrients), con=connection)
        result_nutrientsUnits = pd.read_sql_query(text(selectNutrientsUnits), con=connection)
        result_nutrientsConversion = pd.read_sql_query(text(selectNutrientsConversion), con=connection)
        trans.commit()

    df_food_sources = pd.DataFrame(result_proteinSources)
    df_food_relations = pd.DataFrame(result_proteinRelations)
    df_food_relations.rename(columns={"id":"relation_id"}, inplace=True)
    df_nutrients_composition = pd.DataFrame(result_nutrientsComposition)
    df_nutrients = pd.DataFrame(result_nutrients)
    df_nutrients_units = pd.DataFrame(result_nutrientsUnits)
    df_nutrients_conversion = pd.DataFrame(result_nutrientsConversion)
    df_nutrients_conversion.rename(columns={"nutrient_name":"nutrient"}, inplace=True)

    # Step 2: Process and transform dataset
    #   Get and save informed nutrients
    df_nutrient_conversion_unique = df_nutrients_conversion[['nutrient', 'unit', 'standardized_unit', 'conversion_factor']].drop_duplicates()
    df_nutrient_units_conversion = df_nutrient_conversion_unique.merge(df_nutrients_units, how='left', left_on=['nutrient','unit'], right_on=['nutrient','unit'], suffixes=['','_remove'])
    df_nutrient_units_conversion.drop([i for i in df_nutrient_units_conversion.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns
    df_lookup_nutrients_units_conversion = df_nutrient_units_conversion[~df_nutrient_units_conversion.standardized_unit.isna()]
    df_lookup_nutrients_units_conversion['unit'] = df_lookup_nutrients_units_conversion['unit'].str.replace('µg','ug') # replace micrograms unit symbol

    #   Merge food sources with domains relations
    df_proteins = df_food_sources[~df_food_sources.protein_source.isna()][['id','protein_source_or_food_product', 'protein_source']]
    df_proteins = df_food_sources[['id','protein_source_or_food_product', 'protein_source']]
    df_proteins_relations = df_proteins.merge(df_food_relations, how = 'inner', left_on='id', right_on='food_product_id', suffixes=('', '_remove'))
    df_proteins_relations.drop([i for i in df_proteins_relations.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    #  Merge with nutrients composition
    df_proteins_relations_nutrient_composition = df_proteins_relations.merge(df_nutrients_composition, how='inner', left_on='relation_id', right_on='food_data_source_relation_id', suffixes=('', '_remove'))
    df_proteins_relations_nutrient_composition.drop([i for i in df_proteins_relations_nutrient_composition.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    df_proteins_complete = df_proteins_relations_nutrient_composition.merge(df_nutrients, how='inner', left_on='nutrient_id', right_on='id', suffixes=('', '_remove'))
    df_proteins_complete.drop([i for i in df_proteins_complete.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    # Clean syntax
    df_proteins_complete['amount'] = df_proteins_complete['amount'].fillna('0')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].apply(lambda x: str(x).replace(',','.')) # replace , with . in floating point
    df_proteins_complete['amount'] = df_proteins_complete['amount'].apply(lambda x: str(x).replace('<','')) # replace with nothing cells that contain <
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace('traces', '0.001')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace('1518.61d ', '1518.61')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace('69.33a', '69.33')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace('Not detected', '0')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace(' LOQ', '0')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].astype('float')

    df_proteins_complete['unit'] = df_proteins_complete['unit'].str.replace('µg','ug') # replace micrograms unit symbol

    #   Select nutrients that are informed
    df_proteins_sel = df_proteins_complete[df_proteins_complete.nutrient.isin(df_lookup_nutrients_units_conversion.nutrient)]
    df_proteins_sel = df_proteins_sel[~df_proteins_sel['amount'].isna()]
    
    for idx, row in df_proteins_sel.iterrows():
        # print(row['nutrient'])
        # print(row['unit'])
        df_proteins_sel.loc[idx, 'standardized_unit'] = df_lookup_nutrients_units_conversion.loc[(df_lookup_nutrients_units_conversion['unit'] == row['unit'])&(df_lookup_nutrients_units_conversion['nutrient'] == row['nutrient']), 'standardized_unit'].values[0]
        df_proteins_sel.loc[idx,'standardized_amount'] = df_lookup_nutrients_units_conversion.loc[df_lookup_nutrients_units_conversion['unit'] == row['unit'],'conversion_factor'].values[0]*row['amount']

    if df_proteins_sel.empty:
        #df_proteins_sel = df_proteins_sel.assign(standardized_unit=[], standardized_amount=[])
        df_proteins_sel = pd.DataFrame(columns=['id','protein_source_or_food_product','protein_source'])
        df_proteins_sel['protein_source_or_food_product'] = df_food_sources['protein_source_or_food_product']
        df_proteins_sel['id'] = df_food_sources['id']
        df_proteins_sel['protein_source'] = df_food_sources['protein_source']
        df_proteins_sel = pd.merge(df_proteins_sel, df_nutrients_conversion, how='cross')
        df_proteins_sel = pd.merge(df_proteins_sel, df_nutrients_units, how='inner')
        df_proteins_sel['standardized_amount'] = np.nan
        df_proteins_sel = df_proteins_sel[['id','protein_source_or_food_product','protein_source','nutrient','standardized_unit','standardized_amount']].drop_duplicates()



    df_proteins_sel['nutrient_unit'] = df_proteins_sel['nutrient'] + ' (' + df_proteins_sel['standardized_unit'] + ')' # merge nutrient name and nutrient unit
    df_data = df_proteins_sel[['id','protein_source_or_food_product','protein_source','nutrient_unit', 'standardized_amount']]

    # Multiple reported amounts for the same nutrient and same protein, calculate the mean
    df_data_group = df_data.groupby(['id','protein_source_or_food_product','protein_source','nutrient_unit']).mean().reset_index()
    if df_data_group.empty:
        df_data_group = df_proteins_sel
        df_data_group.loc[df_data_group.protein_source.isna(), 'protein_source'] = 'unknown'
        df_data_group = df_data_group[['id','protein_source_or_food_product','protein_source','nutrient_unit', 'standardized_amount']].groupby(['id','protein_source_or_food_product','protein_source','nutrient_unit']).mean().reset_index()
    # Data is ready to pivot
    df_data_group_pivot = df_data_group.pivot(index=['id','protein_source_or_food_product','protein_source'], columns='nutrient_unit', values='standardized_amount').reset_index()
    #
    df_data_group_pivot.columns.name = ''

    selectNutrientsTheoretical = "SELECT\
                                    *\
                                FROM\
                                    public.nutrients_theoretical_values ntv;"

    with engine.connect() as connection:
        trans = connection.begin()
        result_NutrientsTheoretical = pd.read_sql_query(text(selectNutrientsTheoretical), con=connection)
        trans.commit()

    df_nutrients_theoretical = pd.DataFrame(result_NutrientsTheoretical)
    # nutrient avg
    df_avg_nutrient = df_nutrients_theoretical.describe().transpose().reset_index()
    df_avg_nutrient = df_avg_nutrient.round(2)
    df_avg_nutrient.rename(columns={'index':'nutrient'}, inplace=True)

    # DATA IMPUTATION
    print('Starting nutritional data imputation...')
    # Add variables to input data that are not informed in the database but are in the theoretical nutrients dataset
    added_variables = [col_name for col_name in df_nutrients_theoretical if col_name not in df_data_group_pivot.columns and col_name not in ['Moisture Level', 'Moisture Level Num']]
    #print(f"Added columns: {added_variables}")
    for col in added_variables:
        df_data_group_pivot[col] = np.nan

    #   1 - If the variable value is NA then impute with the nutrient mean value of the same protein source 
    if not df_data_group_pivot['protein_source'].item() == 'unknown':
        for idx, row in df_data_group_pivot.iterrows():
            for jdx, item in row[row.isna()].items():
                df_data_group_pivot.loc[idx,jdx] = df_nutrients_theoretical.loc[df_nutrients_theoretical.protein_source == row['protein_source'], jdx].values[0]

    #   2 - If stills being NA impute with the nutrient mean value
    for idx, row in df_data_group_pivot.iterrows():
        row = row.drop(['id','protein_source', 'protein_source_or_food_product'])
        for jdx, item in row[row.isna()].items():
            df_data_group_pivot.loc[idx,jdx] = df_avg_nutrient.loc[df_avg_nutrient.nutrient == jdx, 'mean'].values[0]
    
    
    return df_data_group_pivot


def getInputProteinAminoAcidData(id, engine):

    # Step 1: Extract raw data from the database
    # Load food sources data
    selectProteinSources = "SELECT \
                                fs2.id,\
                                fs2.protein_source_or_food_product,\
                                psfg.protein_source \
                            FROM \
	                            dcf_data.food_sources fs2 \
                            LEFT JOIN protein_source_format_gm psfg ON fs2.id = psfg.id \
                            WHERE \
	                            fs2.id = '{}'".format(id)

    selectProteinRelations = "SELECT\
                                *\
                            FROM\
                                dcf_data.food_data_source_relations fdsr\
                            WHERE \
                                fdsr.food_product_id = '{}';".format(id)

    selectAminoAcidsComposition = "SELECT\
                                    psfg.protein_source_or_food_product,\
                                    aaf.food_data_source_relation_id,\
                                    aa.amino_acid_name,\
                                    aaf.amino_acid_id,\
                                    aaf.amount,\
                                    aaf.unit\
                                FROM\
                                    protein_source_format_gm psfg\
                                INNER JOIN dcf_data.food_data_source_relations fdsr ON fdsr.food_product_id = psfg.id\
                                INNER JOIN dcf_data.amino_acid_facts aaf ON aaf.food_data_source_relation_id = fdsr.id\
                                INNER JOIN dcf_data.amino_acids aa ON aa.id = aaf.amino_acid_id\
                                INNER JOIN public.aminoacid_units_conversion auc ON (auc.amino_acid_name = aa.amino_acid_name AND auc.unit = aaf.unit)\
                                WHERE auc.conversion_factor IS NOT NULL\
                                AND psfg.id = '{}';".format(id)

    selectAminoAcids = "SELECT \
                            id as amino_acid_id,\
                            amino_acid_name\
                        FROM\
                            dcf_data.amino_acids;"

    selectAminoAcidsUnits = "SELECT DISTINCT\
                                    aa.amino_acid_name,\
                                    aaf.unit\
                                FROM\
                                    dcf_data.amino_acid_facts aaf\
                                INNER JOIN dcf_data.amino_acids aa ON aa.id = aaf.amino_acid_id\
                                INNER JOIN public.aminoacid_units_conversion auc ON (auc.amino_acid_name = aa.amino_acid_name AND auc.unit = aaf.unit)\
                                WHERE auc.conversion_factor IS NOT NULL"

    selectAminoAcidsConversion = "SELECT\
                                    *\
                                FROM\
                                    public.aminoacid_units_conversion auc;"

    selectProteinsWithoutAminoAcids = "SELECT DISTINCT\
                                            ntv.protein_source\
                                        FROM\
                                            nutrients_theoretical_values ntv\
                                        LEFT JOIN aminoacids_theoretical_values atv ON atv.protein_source = ntv.protein_source\
                                        WHERE\
                                            atv.protein_source IS NULL;"

    with engine.connect() as connection:
        trans = connection.begin()
        result_proteinSources = pd.read_sql_query(text(selectProteinSources), con=connection)
        result_proteinRelations = pd.read_sql_query(text(selectProteinRelations), con=connection)
        result_aminoAcidsComposition = pd.read_sql_query(text(selectAminoAcidsComposition), con=connection)
        result_aminoAcids = pd.read_sql_query(text(selectAminoAcids), con=connection)
        result_aminoAcidsUnits = pd.read_sql_query(text(selectAminoAcidsUnits), con=connection)
        result_aminoAcidsConversion = pd.read_sql_query(text(selectAminoAcidsConversion), con=connection)
        result_proteinsWithoutAminoAcids = pd.read_sql_query(text(selectProteinsWithoutAminoAcids), con=connection)
        trans.commit()

    df_food_sources = pd.DataFrame(result_proteinSources)
    df_food_relations = pd.DataFrame(result_proteinRelations)
    df_food_relations.rename(columns={"id":"relation_id"}, inplace=True)
    df_amino_acids_composition = pd.DataFrame(result_aminoAcidsComposition)
    df_amino_acids = pd.DataFrame(result_aminoAcids)
    df_amino_acids_units = pd.DataFrame(result_aminoAcidsUnits)
    df_amino_acids_conversion = pd.DataFrame(result_aminoAcidsConversion)
    df_proteins_without_amino_acids = pd.DataFrame(result_proteinsWithoutAminoAcids)

    #   Merge food sources with domains relations
    # df_proteins = df_food_sources[~df_food_sources.protein_source.isna()][['id','protein_source_or_food_product', 'protein_source']]
    df_proteins = df_food_sources[['id','protein_source_or_food_product', 'protein_source']]
    df_proteins_relations = df_proteins.merge(df_food_relations, how = 'inner', left_on='id', right_on='food_product_id', suffixes=('', '_remove'))
    df_proteins_relations.drop([i for i in df_proteins_relations.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns
    # Merge with aminoacids composition
    df_dataset_aa = df_amino_acids_composition.applymap(lambda x: x.replace('\xa0', ' ').replace('Â', '') if isinstance(x, str) else x)
    df_dataset_aa = df_proteins_relations.merge(df_dataset_aa, how='inner', left_on='relation_id', right_on='food_data_source_relation_id', suffixes=('', '_remove'))
    df_dataset_aa.drop([i for i in df_dataset_aa.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    df_dataset_aa = df_dataset_aa.merge(df_amino_acids, how='right', left_on='amino_acid_id', right_on='amino_acid_id', suffixes=('_remove',''))
    df_dataset_aa.drop([i for i in df_dataset_aa.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns
    # Conversion
    df_dataset_aa.loc[df_dataset_aa.unit.isna(),'unit'] = 'g/100 g Protein'
    df_amino_acids_conversion = df_amino_acids_conversion.applymap(lambda x: x.replace('\xa0', ' ').replace('Â', '') if isinstance(x, str) else x)
    df_dataset_aa_standard = df_dataset_aa.merge(df_amino_acids_conversion, how='left', left_on=['amino_acid_name','unit'], right_on=['amino_acid_name','unit'])
        # If there is no conversion factor or standardized unit, it means that the amount is already standardized and the unit is g/100g Protein, so we fill those values
    df_dataset_aa_standard.loc[df_dataset_aa_standard.standardized_unit.isna(),'standardized_unit'] = 'g/100 g Protein'
    df_dataset_aa_standard.loc[df_dataset_aa_standard.conversion_factor.isna(),'conversion_factor'] = 1
    df_dataset_aa_standard.loc[df_dataset_aa_standard.protein_source_or_food_product.isna(),'protein_source_or_food_product'] = df_proteins.protein_source_or_food_product.values[0]
    df_dataset_aa_standard.loc[df_dataset_aa_standard.id.isna(),'id'] = df_proteins.id.values[0]
    df_dataset_aa_standard.loc[df_dataset_aa_standard.protein_source.isna(),'protein_source'] = df_proteins.protein_source.values[0]

    df_dataset_aa_standard['conversion_factor'] = df_dataset_aa_standard['conversion_factor'].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    df_dataset_aa_standard['amount'] = df_dataset_aa_standard['amount'].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    df_dataset_aa_standard['standardized_amount'] = df_dataset_aa_standard['conversion_factor']*df_dataset_aa_standard['amount']
        # Filter atypical values that are higher than 100g of amino acid per 100g of product
    df_dataset_aa_standard = df_dataset_aa_standard[(df_dataset_aa_standard.standardized_amount <= 100) | (df_dataset_aa_standard.standardized_amount.isna())]

    # Merge essentiality to proteins 
    df_dataset_aa_standard = df_dataset_aa_standard.merge(protscreen_params['df_essentiality'], how='left', left_on='amino_acid_name', right_on='amino_acid_name', suffixes=('', '_remove'))
    df_dataset_aa_standard = df_dataset_aa_standard.merge(protscreen_params['df_source_type'], how='left', left_on= 'protein_source', right_on='protein_source',suffixes=('', '_remove'))
    df_dataset_aa_standard = df_dataset_aa_standard.merge(df_food_sources, how='left', left_on='protein_source_or_food_product', right_on='protein_source_or_food_product', suffixes=('', '_remove'))
    df_dataset_aa_standard.drop([i for i in df_dataset_aa_standard.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    # fill the missing values of protein source and source type with 'unknown' since those are required for the imputation model and they are not informed in the dataset (those that are in the amino acid dataset but not in the nutrients dataset)
    df_dataset_aa_standard.protein_source.fillna('unknown', inplace=True)
    df_dataset_aa_standard.source_type.fillna('unknown', inplace=True)

    df_dataset_aa_standard['amino_acid_unit'] = df_dataset_aa_standard['amino_acid_name'] + ' (' + df_dataset_aa_standard['standardized_unit'] + ')' # merge nutrient name and nutrient unit
    df_data = df_dataset_aa_standard[['id','protein_source_or_food_product','protein_source','source_type','amino_acid_unit', 'standardized_amount']]

    # Multiple reported amounts for the same nutrient and same protein, calculate the mean
    df_data_group = df_data[['id','protein_source_or_food_product','protein_source','source_type','amino_acid_unit','standardized_amount']].groupby(['id','protein_source_or_food_product','protein_source','source_type','amino_acid_unit']).mean().reset_index()
    # Data is ready to pivot
    df_data_group_pivot = df_data_group.pivot(index=['id','protein_source_or_food_product','protein_source','source_type'], columns='amino_acid_unit', values='standardized_amount').reset_index()
    df_aminoacid_unit_list = pd.DataFrame(data={'amino_acid_unit':df_data_group_pivot.columns[5:]})

    df_data_group_pivot.columns.name = ''

    # # aminoacids for each protein source
    # df_data_group_pivot = df_data_group_pivot.drop(columns=['source_type'])
    # df_avg_aminoacid_protein_source = df_data_group_pivot.iloc[:,1:].groupby('protein_source').mean().reset_index()
    
    # for variable in df_avg_aminoacid_protein_source.columns[1:]:
    #     var_mean = df_avg_aminoacid_protein_source[variable].mean()
    #     df_avg_aminoacid_protein_source[variable] = df_avg_aminoacid_protein_source[variable].fillna(var_mean)

    #     # Imputa mean values to protein sources without amino acid data (those that are in the nutrients dataset but not in the amino acid dataset)
    # for idx, row in df_proteins_without_amino_acids.iterrows():
    #     df_avg_aminoacid_protein_source.loc[len(df_avg_aminoacid_protein_source),'protein_source'] = row['protein_source'] 
    #     for variable in df_avg_aminoacid_protein_source.columns[1:]:
    #         var_mean = df_avg_aminoacid_protein_source[variable].mean()
    #         df_avg_aminoacid_protein_source.loc[df_avg_aminoacid_protein_source['protein_source'] == row['protein_source'], variable] = var_mean

    selectAminoAcidsTheoretical = "SELECT\
                                    *\
                                FROM\
                                    public.aminoacids_theoretical_values ntv;"

    with engine.connect() as connection:
        trans = connection.begin()
        result_AminoAcidsTheoretical = pd.read_sql_query(text(selectAminoAcidsTheoretical), con=connection)
        trans.commit()

    df_aminoacids_theoretical = pd.DataFrame(result_AminoAcidsTheoretical)
    # aminoacid avg
    df_avg_aminoacid = df_aminoacids_theoretical.describe().transpose().reset_index()
    df_avg_aminoacid = df_avg_aminoacid.round(2)
    df_avg_aminoacid.rename(columns={'index':'amino_acid'}, inplace=True)

    # DATA IMPUTATION
    print('Starting amino acid data imputation...')
    # Select only amino acid theoretical value columns
    sel_columns = ['id', 'source_type']
    sel_columns.extend(list(df_aminoacids_theoretical.columns))
    df_data_group_pivot = df_data_group_pivot[sel_columns]
    #   1 - If the variable value is NA then impute with the amino acid mean value of the same protein source
    if not df_data_group_pivot['protein_source'].item() == 'unknown':
        for idx, row in df_data_group_pivot.iterrows():
            for jdx, item in row[row.isna()].items():
                df_data_group_pivot.loc[idx,jdx] = df_aminoacids_theoretical.loc[df_aminoacids_theoretical.protein_source == row['protein_source'], jdx].values[0]

    #   2 - If stills being NA impute with the amino acid mean value
    for idx, row in df_data_group_pivot.iterrows():
        for jdx, item in row[row.isna()].items():
            df_data_group_pivot.loc[idx,jdx] = df_avg_aminoacid.loc[df_avg_aminoacid.amino_acid == jdx, 'mean'].values[0]

    return df_data_group_pivot


def getInputProteinEnvironmentalData(id, engine):

    # Step 1: Extract raw data from the database
    # Load food sources data
    selectProteinSources = "SELECT \
                                fs2.id,\
                                fs2.protein_source_or_food_product,\
                                psfg.protein_source \
                            FROM \
	                            dcf_data.food_sources fs2 \
                            LEFT JOIN protein_source_format_gm psfg ON fs2.id = psfg.id \
                            WHERE \
	                            fs2.id = '{}'".format(id)

    selectProteinRelations = "SELECT\
                                *\
                            FROM\
                                dcf_data.food_data_source_relations fdsr\
                            WHERE \
                                fdsr.food_product_id = '{}';".format(id)

    selectEnvironmentalImpact = "SELECT\
                                    psfg.protein_source_or_food_product,\
                                    cia.food_data_source_relation_id,\
                                    eei.indicator_name,\
                                    eeif.env_indicator_id,\
                                    eeif.amount_or_impact_value,\
                                    eeif.unit\
                                FROM\
                                    protein_source_format_gm psfg\
                                INNER JOIN dcf_data.food_data_source_relations fdsr ON fdsr.food_product_id = psfg.id\
                                INNER JOIN dcf_data.combined_inventory_activities cia on cia.food_data_source_relation_id = fdsr.id\
                                INNER JOIN dcf_data.elca_environment_indicator_facts eeif on cia.id = eeif.activity_id\
                                INNER JOIN dcf_data.elca_environment_indicators eei ON eei.id = eeif.env_indicator_id\
                                WHERE\
                                    psfg.id = '{}';".format(id)

    selectEnvironmentalIndicators = "select\
                                        *\
                                    from\
                                        dcf_data.elca_environment_indicators eei;"

    selectEnvironmentalUnits = "SELECT DISTINCT\
                                    eei.indicator_name,\
                                    eeif.unit\
                                FROM\
                                    dcf_data.elca_environment_indicator_facts eeif\
                                INNER JOIN dcf_data.elca_environment_indicators eei ON eei.id = eeif.env_indicator_id;"

    selectProteinsWithoutEnvironmentalValues = "SELECT DISTINCT\
                                            ntv.protein_source\
                                        FROM\
                                            nutrients_theoretical_values ntv\
                                        LEFT JOIN environmental_theoretical_values etv on etv.protein_source = ntv.protein_source\
                                        WHERE\
                                            etv.protein_source IS NULL;"

    with engine.connect() as connection:
        trans = connection.begin()
        result_proteinSources = pd.read_sql_query(text(selectProteinSources), con=connection)
        result_proteinRelations = pd.read_sql_query(text(selectProteinRelations), con=connection)
        result_environmentalImpact = pd.read_sql_query(text(selectEnvironmentalImpact), con=connection)
        result_environmentalIndicators = pd.read_sql_query(text(selectEnvironmentalIndicators), con=connection)
        result_environmentalUnits = pd.read_sql_query(text(selectEnvironmentalUnits), con=connection)
        result_proteinsWithoutEnvironmentalValues = pd.read_sql_query(text(selectProteinsWithoutEnvironmentalValues), con=connection)
        trans.commit()

    df_food_sources = pd.DataFrame(result_proteinSources)
    df_food_relations = pd.DataFrame(result_proteinRelations)
    df_food_relations.rename(columns={"id":"relation_id"}, inplace=True)
    df_environmental_impact = pd.DataFrame(result_environmentalImpact)
    df_environmental_indicators = pd.DataFrame(result_environmentalIndicators)
    df_environmental_units = pd.DataFrame(result_environmentalUnits)
    df_proteins_without_environmental_values = pd.DataFrame(result_proteinsWithoutEnvironmentalValues)

    #   Merge food sources with domains relations
    # df_proteins = df_food_sources[~df_food_sources.protein_source.isna()][['id','protein_source_or_food_product', 'protein_source']]
    df_proteins = df_food_sources[['id','protein_source_or_food_product', 'protein_source']]
    df_proteins_relations = df_proteins.merge(df_food_relations, how = 'inner', left_on='id', right_on='food_product_id', suffixes=('', '_remove'))
    df_proteins_relations.drop([i for i in df_proteins_relations.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns
    # Merge with environmental impact values
    df_proteins_relations = df_proteins_relations.merge(df_environmental_impact, how='left', left_on='relation_id', right_on='food_data_source_relation_id', suffixes=('', '_remove'))
    df_proteins_relations.drop([i for i in df_proteins_relations.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    df_dataset_env = df_proteins_relations.merge(df_environmental_units, how='cross', suffixes=('_remove',''))
    df_dataset_env.drop([i for i in df_dataset_env.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    #df_env_indicators_units = df_environmental_indicators.merge(df_environmental_units, how='inner', left_on='indicator_name', right_on='indicator_name')

    df_dataset_env.loc[df_dataset_env.protein_source_or_food_product.isna(),'protein_source_or_food_product'] = df_proteins.protein_source_or_food_product.values[0]
    df_dataset_env.loc[df_dataset_env.id.isna(),'id'] = df_proteins.id.values[0]
    df_dataset_env.loc[df_dataset_env.protein_source.isna(),'protein_source'] = df_proteins.protein_source.values[0]

    df_dataset_env = df_dataset_env.merge(df_environmental_units, how='inner', left_on='indicator_name', right_on='indicator_name', suffixes=('_remove',''))
    df_dataset_env.drop([i for i in df_dataset_env.columns if 'remove' in i], axis=1, inplace=True)

    # Conversion

    # df_amino_acids_conversion = df_amino_acids_conversion.applymap(lambda x: x.replace('\xa0', ' ').replace('Â', '') if isinstance(x, str) else x)
    # df_dataset_aa_standard = df_dataset_aa.merge(df_amino_acids_conversion, how='left', left_on=['amino_acid_name','unit'], right_on=['amino_acid_name','unit'])
    #     # If there is no conversion factor or standardized unit, it means that the amount is already standardized and the unit is g/100g Protein, so we fill those values
    # df_dataset_aa_standard.loc[df_dataset_aa_standard.standardized_unit.isna(),'standardized_unit'] = 'g/100 g Protein'
    # df_dataset_aa_standard.loc[df_dataset_aa_standard.conversion_factor.isna(),'conversion_factor'] = 1
    # df_dataset_aa_standard.loc[df_dataset_aa_standard.protein_source_or_food_product.isna(),'protein_source_or_food_product'] = df_proteins.protein_source_or_food_product.values[0]
    # df_dataset_aa_standard.loc[df_dataset_aa_standard.id.isna(),'id'] = df_proteins.id.values[0]
    # df_dataset_aa_standard.loc[df_dataset_aa_standard.protein_source.isna(),'protein_source'] = df_proteins.protein_source.values[0]

    # df_dataset_aa_standard['conversion_factor'] = df_dataset_aa_standard['conversion_factor'].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    # df_dataset_aa_standard['amount'] = df_dataset_aa_standard['amount'].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    # df_dataset_aa_standard['standardized_amount'] = df_dataset_aa_standard['conversion_factor']*df_dataset_aa_standard['amount']
    #     # Filter atypical values that are higher than 100g of amino acid per 100g of product
    # df_dataset_aa_standard = df_dataset_aa_standard[(df_dataset_aa_standard.standardized_amount <= 100) | (df_dataset_aa_standard.standardized_amount.isna())]

    # # Merge essentiality to proteins 
    # df_dataset_aa_standard = df_dataset_aa_standard.merge(protscreen_params['df_essentiality'], how='left', left_on='amino_acid_name', right_on='amino_acid_name', suffixes=('', '_remove'))
    # df_dataset_aa_standard = df_dataset_aa_standard.merge(protscreen_params['df_source_type'], how='left', left_on= 'protein_source', right_on='protein_source',suffixes=('', '_remove'))
    # df_dataset_aa_standard = df_dataset_aa_standard.merge(df_food_sources, how='left', left_on='protein_source_or_food_product', right_on='protein_source_or_food_product', suffixes=('', '_remove'))
    # df_dataset_aa_standard.drop([i for i in df_dataset_aa_standard.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    # fill the missing values of protein source and source type with 'unknown' since those are required for the imputation model and they are not informed in the dataset (those that are in the amino acid dataset but not in the nutrients dataset)
    df_dataset_env.protein_source.fillna('unknown', inplace=True)
    #df_dataset_env.source_type.fillna('unknown', inplace=True)

    df_dataset_env['indicator_unit'] = df_dataset_env['indicator_name'] + ' (' + df_dataset_env['unit'] + ')' # merge environmental name and environmental unit
    df_data = df_dataset_env[['id','protein_source_or_food_product','protein_source','indicator_unit', 'amount_or_impact_value']]

    # Multiple reported amounts for the same nutrient and same protein, calculate the mean
    df_data_group = df_data[['id','protein_source_or_food_product','protein_source','indicator_unit','amount_or_impact_value']].groupby(['id','protein_source_or_food_product','protein_source','indicator_unit']).mean().reset_index()
    # Data is ready to pivot
    df_data_group_pivot = df_data_group.pivot(index=['id','protein_source_or_food_product','protein_source'], columns='indicator_unit', values='amount_or_impact_value').reset_index()
    df_environmental_unit_list = pd.DataFrame(data={'indicator_unit':df_data_group_pivot.columns[5:]})

    df_data_group_pivot.columns.name = ''

    # # aminoacids for each protein source
    # df_data_group_pivot = df_data_group_pivot.drop(columns=['source_type'])
    # df_avg_aminoacid_protein_source = df_data_group_pivot.iloc[:,1:].groupby('protein_source').mean().reset_index()
    
    # for variable in df_avg_aminoacid_protein_source.columns[1:]:
    #     var_mean = df_avg_aminoacid_protein_source[variable].mean()
    #     df_avg_aminoacid_protein_source[variable] = df_avg_aminoacid_protein_source[variable].fillna(var_mean)

    #     # Imputa mean values to protein sources without amino acid data (those that are in the nutrients dataset but not in the amino acid dataset)
    # for idx, row in df_proteins_without_amino_acids.iterrows():
    #     df_avg_aminoacid_protein_source.loc[len(df_avg_aminoacid_protein_source),'protein_source'] = row['protein_source'] 
    #     for variable in df_avg_aminoacid_protein_source.columns[1:]:
    #         var_mean = df_avg_aminoacid_protein_source[variable].mean()
    #         df_avg_aminoacid_protein_source.loc[df_avg_aminoacid_protein_source['protein_source'] == row['protein_source'], variable] = var_mean

    selectEnvironmentalTheoretical = "SELECT\
                                    *\
                                FROM\
                                    public.environmental_theoretical_values ntv;"

    with engine.connect() as connection:
        trans = connection.begin()
        result_EnvironmentalTheoretical = pd.read_sql_query(text(selectEnvironmentalTheoretical), con=connection)
        trans.commit()

    df_environmental_theoretical = pd.DataFrame(result_EnvironmentalTheoretical)
    # environmental avg
    df_avg_environmental = df_environmental_theoretical.describe().transpose().reset_index()
    df_avg_environmental = df_avg_environmental.round(2)
    df_avg_environmental.rename(columns={'index':'environmental_variable'}, inplace=True)

    # DATA IMPUTATION
    print('Starting environmental data imputation...')
    # Select only environmental theoretical value columns
    sel_columns = ['id']
    sel_columns.extend(list(df_environmental_theoretical.columns))
    df_data_group_pivot = df_data_group_pivot[sel_columns]
    df_data_group_pivot[list(df_environmental_theoretical.columns)[1:]] = df_data_group_pivot[list(df_environmental_theoretical.columns)[1:]].astype(float)
    #   1 - If the variable value is NA then impute with the environmental variable mean value of the same protein source
    if not df_data_group_pivot['protein_source'].item() == 'unknown':
        for idx, row in df_data_group_pivot.iterrows():
            for jdx, item in row[row.isna()].items():
                df_data_group_pivot.loc[idx,jdx] = df_environmental_theoretical.loc[df_environmental_theoretical.protein_source == row['protein_source'], jdx].values[0]

    #   2 - If stills being NA impute with the environmental variable mean value
    for idx, row in df_data_group_pivot.iterrows():
        for jdx, item in row[row.isna()].items():
            df_data_group_pivot.loc[idx,jdx] = df_avg_environmental.loc[df_avg_environmental.environmental_variable == jdx, 'mean'].values[0]

    return df_data_group_pivot


def generateCleanDatasetNutrients(engine):
    ##################################################
    # NUTRITIONAL DATASET CLEANING AND TRANSFORMATION
    ##################################################
    print('Starting nutritional dataset cleaning and transformation...')
    # Step 1: Extract raw data from the database
    # Load food sources data
    selectProteinSources = "SELECT\
                            *\
                        FROM\
	                        public.protein_source_format_gm psfg"

    selectProteinRelations = "SELECT\
                                *\
                            FROM\
                                dcf_data.food_data_source_relations fdsr;"

    selectNutrientsComposition = "SELECT\
                                    psfg.protein_source_or_food_product,\
                                    ncf.food_data_source_relation_id,\
                                    n.id AS \"nutrient_id\",\
                                    n.nutrient,\
                                    ncf.amount AS \"amount\",\
                                    ncf.unit AS \"unit\"\
                                FROM\
                                    public.protein_source_format_gm psfg\
                                INNER JOIN dcf_data.food_data_source_relations fdsr ON fdsr.food_product_id = psfg.id\
                                INNER JOIN dcf_data.nutrient_composition_facts ncf ON ncf.food_data_source_relation_id = fdsr.id \
                                INNER JOIN dcf_data.nutrients n ON n.id = ncf.nutrient_id\
                                INNER JOIN public.nutrients_units_conversion nuc ON (nuc.nutrient_name = n.nutrient AND nuc.unit = ncf.unit)\
                                WHERE nuc.conversion_factor IS NOT NULL;"

    selectNutrients = "SELECT\
                            *\
                        FROM\
                            dcf_data.nutrients n;"

    selectNutrientsUnits = "SELECT DISTINCT\
                                    n.nutrient,\
                                    ncf.unit\
                                FROM\
                                    dcf_data.nutrient_composition_facts ncf\
                                INNER JOIN dcf_data.nutrients n ON n.id = ncf.nutrient_id\
                                INNER JOIN public.nutrients_units_conversion nuc ON (nuc.nutrient_name = n.nutrient AND nuc.unit = ncf.unit)\
                                WHERE nuc.conversion_factor IS NOT NULL"

    selectNutrientsConversion = "SELECT\
                                    *\
                                FROM\
                                    public.nutrients_units_conversion nc;"


    with engine.connect() as connection:
        trans = connection.begin()
        result_proteinSources = pd.read_sql_query(text(selectProteinSources), con=connection)
        trans.commit()

    with engine.connect() as connection:
        trans = connection.begin()
        result_proteinSources = pd.read_sql_query(text(selectProteinSources), con=connection)
        result_proteinRelations = pd.read_sql_query(text(selectProteinRelations), con=connection)
        result_nutrientsComposition = pd.read_sql_query(text(selectNutrientsComposition), con=connection)
        result_nutrients = pd.read_sql_query(text(selectNutrients), con=connection)
        result_nutrientsUnits = pd.read_sql_query(text(selectNutrientsUnits), con=connection)
        result_nutrientsConversion = pd.read_sql_query(text(selectNutrientsConversion), con=connection)
        trans.commit()

    df_food_sources = pd.DataFrame(result_proteinSources)
    df_food_relations = pd.DataFrame(result_proteinRelations)
    df_food_relations.rename(columns={"id":"relation_id"}, inplace=True)
    df_nutrients_composition = pd.DataFrame(result_nutrientsComposition)
    df_nutrients = pd.DataFrame(result_nutrients)
    df_nutrients_units = pd.DataFrame(result_nutrientsUnits)
    df_nutrients_conversion = pd.DataFrame(result_nutrientsConversion)
    df_nutrients_conversion.rename(columns={"nutrient_name":"nutrient"}, inplace=True)

    # Step 2: Process and transform dataset
    #   Get and save informed nutrients
    df_nutrient_conversion_unique = df_nutrients_conversion[['nutrient', 'unit', 'standardized_unit', 'conversion_factor']].drop_duplicates()
    df_nutrient_units_conversion = df_nutrient_conversion_unique.merge(df_nutrients_units, how='left', left_on=['nutrient','unit'], right_on=['nutrient','unit'], suffixes=['','_remove'])
    df_nutrient_units_conversion.drop([i for i in df_nutrient_units_conversion.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns
    df_lookup_nutrients_units_conversion = df_nutrient_units_conversion[~df_nutrient_units_conversion.standardized_unit.isna()]
    df_lookup_nutrients_units_conversion['unit'] = df_lookup_nutrients_units_conversion['unit'].str.replace('µg','ug') # replace micrograms unit symbol

    #   Merge food sources with domains relations
    df_proteins = df_food_sources[~df_food_sources.protein_source.isna()][['id','protein_source_or_food_product', 'protein_source']]
    df_proteins_relations = df_proteins.merge(df_food_relations, how = 'inner', left_on='id', right_on='food_product_id', suffixes=('', '_remove'))
    df_proteins_relations.drop([i for i in df_proteins_relations.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    #  Merge with nutrients composition
    df_proteins_relations_nutrient_composition = df_proteins_relations.merge(df_nutrients_composition, how='inner', left_on='relation_id', right_on='food_data_source_relation_id', suffixes=('', '_remove'))
    df_proteins_relations_nutrient_composition.drop([i for i in df_proteins_relations_nutrient_composition.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    df_proteins_complete = df_proteins_relations_nutrient_composition.merge(df_nutrients, how='inner', left_on='nutrient_id', right_on='id', suffixes=('', '_remove'))
    df_proteins_complete.drop([i for i in df_proteins_complete.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    # Clean syntax
    df_proteins_complete['amount'] = df_proteins_complete['amount'].fillna('0')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].apply(lambda x: str(x).replace(',','.')) # replace , with . in floating point
    df_proteins_complete['amount'] = df_proteins_complete['amount'].apply(lambda x: str(x).replace('<','')) # replace with nothing cells that contain <
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace('traces', '0.001')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace('1518.61d ', '1518.61')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace('69.33a', '69.33')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace('92.6 (84.5)', '92.6')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace('94.4 (86.1)', '94.4')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace('Not detected', '0')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].replace('LOQ', '0')
    df_proteins_complete['amount'] = df_proteins_complete['amount'].astype('float')

    df_proteins_complete['unit'] = df_proteins_complete['unit'].str.replace('µg','ug') # replace micrograms unit symbol

    #   Select nutrients that are informed
    df_proteins_sel = df_proteins_complete[df_proteins_complete.nutrient.isin(df_lookup_nutrients_units_conversion.nutrient)]
    df_proteins_sel = df_proteins_sel[~df_proteins_sel['amount'].isna()]
    
    for idx, row in df_proteins_sel.iterrows():
        # print(row['nutrient'])
        # print(row['unit'])
        df_proteins_sel.loc[idx, 'standardized_unit'] = df_lookup_nutrients_units_conversion.loc[(df_lookup_nutrients_units_conversion['unit'] == row['unit'])&(df_lookup_nutrients_units_conversion['nutrient'] == row['nutrient']), 'standardized_unit'].values[0]
        df_proteins_sel.loc[idx,'standardized_amount'] = df_lookup_nutrients_units_conversion.loc[df_lookup_nutrients_units_conversion['unit'] == row['unit'],'conversion_factor'].values[0]*row['amount']

    df_proteins_sel['nutrient_unit'] = df_proteins_sel['nutrient'] + ' (' + df_proteins_sel['standardized_unit'] + ')' # merge nutrient name and nutrient unit
    df_data = df_proteins_sel[['protein_source_or_food_product','protein_source','nutrient_unit', 'standardized_amount']]

    # Multiple reported amounts for the same nutrient and same protein, calculate the mean
    df_data_group = df_data[['protein_source_or_food_product','protein_source','nutrient_unit','standardized_amount']].groupby(['protein_source_or_food_product','protein_source','nutrient_unit']).mean().reset_index()
    # Data is ready to pivot
    df_data_group_pivot = df_data_group.pivot(index=['protein_source_or_food_product','protein_source'], columns='nutrient_unit', values='standardized_amount').reset_index()
    #
    df_data_group_pivot.columns.name = ''

    # MOISTURE COMPUTATION AND CLASSIFICATION
    df_data_group_pivot = df_data_group_pivot.apply(moistureComputationClassification, axis=1)

    # CALCULATES AVG VALUES OF NUTRIENTS PER PROTEIN SOURCE AND AVG NUTRIENT VALUES
    # nutrients for each protein source
    df_avg_nutrient_protein_source = df_data_group_pivot.iloc[:,1:].groupby(['protein_source', 'Moisture Level', 'Moisture Level Num']).mean().reset_index()
    df_avg_nutrient_protein_source = df_avg_nutrient_protein_source.round(2) 
    # save nutritional theoretical values per protein source
    df_avg_nutrient_protein_source.to_sql('nutrients_theoretical_values', engine, if_exists='replace', index=False)
    # nutrient avg
    df_avg_nutrient = df_data_group_pivot.describe().transpose().reset_index()
    df_avg_nutrient = df_avg_nutrient.round(2)
    df_avg_nutrient.rename(columns={'':'nutrient'}, inplace=True)

    # DATA IMPUTATION
    print('Starting nutritional data imputation...')
    #   1 - If the variable value is NA then impute with the nutrient mean value of the same protein source
    moisture_distance_dict = {'Very Low': list(['Low', 'Intermediate', 'High', 'Very High']), 'Low': list(['Very Low', 'Intermediate', 'High', 'Very High']), 'Intermediate': list(['Low', 'High', 'Very Low', 'Very High']), 'High': list(['Intermediate', 'Very High', 'Low', 'Very Low']), 'Very High': list(['High', 'Intermediate', 'Low', 'Very Low'])}
    for idx, row in df_data_group_pivot.iterrows():
        print(row['protein_source_or_food_product'])
        for jdx, item in row[row.isna()].items():
            print(jdx)
            if df_avg_nutrient_protein_source.loc[(df_avg_nutrient_protein_source.protein_source == row['protein_source']) & (df_avg_nutrient_protein_source['Moisture Level'] == row['Moisture Level']), jdx].empty:
                # if there are no avg values for that protein source and moisture level, find the closest below moisture level
                for moisture_level in moisture_distance_dict[row['Moisture Level']]:
                    if not df_avg_nutrient_protein_source.loc[(df_avg_nutrient_protein_source.protein_source == row['protein_source']) & (df_avg_nutrient_protein_source['Moisture Level'] == moisture_level), jdx].empty:
                        df_data_group_pivot.loc[idx,jdx] = df_avg_nutrient_protein_source.loc[(df_avg_nutrient_protein_source.protein_source == row['protein_source']) & (df_avg_nutrient_protein_source['Moisture Level'] == moisture_level), jdx].values[0]
                        break
            else:
                df_data_group_pivot.loc[idx,jdx] = df_avg_nutrient_protein_source.loc[(df_avg_nutrient_protein_source.protein_source == row['protein_source']) & (df_avg_nutrient_protein_source['Moisture Level'] == row['Moisture Level']), jdx].values[0]

    #   2 - If stills being NA impute with the nutrient mean value
    for idx, row in df_data_group_pivot.iterrows():
        for jdx, item in row[row.isna()].items():
            df_data_group_pivot.loc[idx,jdx] = df_avg_nutrient.loc[df_avg_nutrient.nutrient == jdx, 'mean'].values[0]

    # SAVE CLEAN DATAFRAME TO THE DATABASE
    df_data_group_pivot.to_sql('nutrients_data_imputation_model_input', engine, if_exists='replace', index=False)

    # NORMALISATION MIN-MAX OVER THE CLEAN DATASET
    print('Starting nutritional data normalisation...')
    df_final_nutrients_scaled = df_data_group_pivot.copy()

    for column in df_final_nutrients_scaled.columns[2:len(df_final_nutrients_scaled.columns)-2]:
        df_final_nutrients_scaled[column] = (df_final_nutrients_scaled[column] - df_final_nutrients_scaled[column].min()) / (df_final_nutrients_scaled[column].max() - df_final_nutrients_scaled[column].min())

    df_final_nutrients_scaled.to_sql('nutrients_data_imputation_model_input_normalised', engine, if_exists='replace', index=False)

    return

def generateCleanDatasetAminoAcids(engine):
    ###################################################
    #   AMINO ACID DATASET CLEANING AND TRANSFORMATION
    ###################################################
    print('Starting amino acid dataset cleaning and transformation...')
    # Step 1: Extract raw data from the database
    # Load food sources data
    selectProteinSources = "SELECT\
                            *\
                        FROM\
	                        public.protein_source_format_gm psfg"

    selectProteinRelations = "SELECT\
                                *\
                            FROM\
                                dcf_data.food_data_source_relations fdsr;"

    selectAminoAcidsComposition = "SELECT\
                                    psfg.protein_source_or_food_product,\
                                    aaf.food_data_source_relation_id,\
                                    aa.amino_acid_name,\
                                    aaf.amino_acid_id,\
                                    aaf.amount,\
                                    aaf.unit\
                                FROM\
                                    protein_source_format_gm psfg\
                                INNER JOIN dcf_data.food_data_source_relations fdsr ON fdsr.food_product_id = psfg.id\
                                INNER JOIN dcf_data.amino_acid_facts aaf ON aaf.food_data_source_relation_id = fdsr.id\
                                INNER JOIN dcf_data.amino_acids aa ON aa.id = aaf.amino_acid_id\
                                INNER JOIN public.aminoacid_units_conversion auc ON (auc.amino_acid_name = aa.amino_acid_name AND auc.unit = aaf.unit)\
                                WHERE auc.conversion_factor IS NOT NULL;"

    selectAminoAcids = "SELECT\
                            *\
                        FROM\
                            dcf_data.amino_acids;"

    selectAminoAcidsUnits = "SELECT DISTINCT\
                                    aa.amino_acid_name,\
                                    aaf.unit\
                                FROM\
                                    dcf_data.amino_acid_facts aaf\
                                INNER JOIN dcf_data.amino_acids aa ON aa.id = aaf.amino_acid_id\
                                INNER JOIN public.aminoacid_units_conversion auc ON (auc.amino_acid_name = aa.amino_acid_name AND auc.unit = aaf.unit)\
                                WHERE auc.conversion_factor IS NOT NULL"

    selectAminoAcidsConversion = "SELECT\
                                    *\
                                FROM\
                                    public.aminoacid_units_conversion auc;"

    selectProteinsWithoutAminoAcids = "SELECT DISTINCT\
                                            ntv.protein_source\
                                        FROM\
                                            nutrients_theoretical_values ntv\
                                        LEFT JOIN aminoacids_theoretical_values atv ON atv.protein_source = ntv.protein_source\
                                        WHERE\
                                            atv.protein_source IS NULL;"

    with engine.connect() as connection:
        trans = connection.begin()
        result_proteinSources = pd.read_sql_query(text(selectProteinSources), con=connection)
        result_proteinRelations = pd.read_sql_query(text(selectProteinRelations), con=connection)
        result_aminoAcidsComposition = pd.read_sql_query(text(selectAminoAcidsComposition), con=connection)
        result_aminoAcids = pd.read_sql_query(text(selectAminoAcids), con=connection)
        result_aminoAcidsUnits = pd.read_sql_query(text(selectAminoAcidsUnits), con=connection)
        result_aminoAcidsConversion = pd.read_sql_query(text(selectAminoAcidsConversion), con=connection)
        result_proteinsWithoutAminoAcids = pd.read_sql_query(text(selectProteinsWithoutAminoAcids), con=connection)
        trans.commit()

    df_food_sources = pd.DataFrame(result_proteinSources)
    df_food_relations = pd.DataFrame(result_proteinRelations)
    df_food_relations.rename(columns={"id":"relation_id"}, inplace=True)
    df_amino_acids_composition = pd.DataFrame(result_aminoAcidsComposition)
    df_amino_acids = pd.DataFrame(result_aminoAcids)
    df_amino_acids_units = pd.DataFrame(result_aminoAcidsUnits)
    df_amino_acids_conversion = pd.DataFrame(result_aminoAcidsConversion)
    df_proteins_without_amino_acids = pd.DataFrame(result_proteinsWithoutAminoAcids)

    #   Merge food sources with domains relations
    df_proteins = df_food_sources[~df_food_sources.protein_source.isna()][['id','protein_source_or_food_product', 'protein_source']]
    df_proteins_relations = df_proteins.merge(df_food_relations, how = 'inner', left_on='id', right_on='food_product_id', suffixes=('', '_remove'))
    df_proteins_relations.drop([i for i in df_proteins_relations.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns
    # Merge with aminoacids composition
    df_dataset_aa = df_amino_acids_composition.applymap(lambda x: x.replace('\xa0', ' ').replace('Â', '') if isinstance(x, str) else x)
    df_dataset_aa = df_proteins_relations.merge(df_dataset_aa, how='left', left_on='relation_id', right_on='food_data_source_relation_id', suffixes=('', '_remove'))
    df_dataset_aa.drop([i for i in df_dataset_aa.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    df_dataset_aa = df_dataset_aa.merge(df_amino_acids, how='inner', left_on='amino_acid_id', right_on='id', suffixes=('', '_remove'))
    df_dataset_aa.drop([i for i in df_dataset_aa.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns
    # Conversion
    df_dataset_aa_standard = df_dataset_aa.merge(df_amino_acids_conversion, how='inner', left_on=['amino_acid_name','unit'], right_on=['amino_acid_name','unit'])
    df_dataset_aa_standard['conversion_factor'] = df_dataset_aa_standard['conversion_factor'].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    df_dataset_aa_standard['amount'] = df_dataset_aa_standard['amount'].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    df_dataset_aa_standard['standardized_amount'] = df_dataset_aa_standard['conversion_factor']*df_dataset_aa_standard['amount']
        # Filter atypical values that are higher than 100g of amino acid per 100g of product
    df_dataset_aa_standard = df_dataset_aa_standard[df_dataset_aa_standard.standardized_amount <= 100]

    # Merge essentiality to proteins 
    df_dataset_aa_standard = df_dataset_aa_standard.merge(protscreen_params['df_essentiality'], how='inner', left_on='amino_acid_name', right_on='amino_acid_name', suffixes=('', '_remove'))
    df_dataset_aa_standard = df_dataset_aa_standard.merge(protscreen_params['df_source_type'], how='inner', left_on= 'protein_source', right_on='protein_source',suffixes=('', '_remove'))
    df_dataset_aa_standard = df_dataset_aa_standard.merge(df_food_sources, how='inner', left_on='protein_source_or_food_product', right_on='protein_source_or_food_product', suffixes=('', '_remove'))
    df_dataset_aa_standard.drop([i for i in df_dataset_aa_standard.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    df_dataset_aa_standard['amino_acid_unit'] = df_dataset_aa_standard['amino_acid_name'] + ' (' + df_dataset_aa_standard['standardized_unit'] + ')' # merge nutrient name and nutrient unit
    df_data = df_dataset_aa_standard[['protein_source_or_food_product','protein_source','category','source_type','amino_acid_unit', 'standardized_amount']]

    # Multiple reported amounts for the same nutrient and same protein, calculate the mean
    df_data_group = df_data[['protein_source_or_food_product','protein_source','category','source_type','amino_acid_unit','standardized_amount']].groupby(['protein_source_or_food_product','protein_source','category','source_type','amino_acid_unit']).mean().reset_index()
    # Data is ready to pivot
    df_data_group_pivot = df_data_group.pivot(index=['protein_source_or_food_product','protein_source','category','source_type'], columns='amino_acid_unit', values='standardized_amount').reset_index()
    df_aminoacid_unit_list = pd.DataFrame(data={'amino_acid_unit':df_data_group_pivot.columns[4:]})

    df_data_group_pivot.columns.name = ''

    # aminoacids for each protein source
    df_data_group_pivot = df_data_group_pivot.drop(columns=['category', 'source_type'])
    df_avg_aminoacid_protein_source = df_data_group_pivot.iloc[:,1:].groupby('protein_source').mean().reset_index()
    
    for variable in df_avg_aminoacid_protein_source.columns[1:]:
        var_mean = df_avg_aminoacid_protein_source[variable].mean()
        df_avg_aminoacid_protein_source[variable] = df_avg_aminoacid_protein_source[variable].fillna(var_mean)

    # Imputa mean values to protein sources without amino acid data (those that are in the nutrients dataset but not in the amino acid dataset)
    for idx, row in df_proteins_without_amino_acids.iterrows():
        df_avg_aminoacid_protein_source.loc[len(df_avg_aminoacid_protein_source),'protein_source'] = row['protein_source'] 
        for variable in df_avg_aminoacid_protein_source.columns[1:]:
            var_mean = df_avg_aminoacid_protein_source[variable].mean()
            df_avg_aminoacid_protein_source.loc[df_avg_aminoacid_protein_source['protein_source'] == row['protein_source'], variable] = var_mean
    
    # !!! TO IMPROVE: For some reason there are some protein sources that are in the nutrients dataset but not in the amino acid dataset and they are not being captured by the query above. So, I will check for those protein sources and impute their values as well.
    proteins_without_aminoacids = [elemento for elemento in list(df_proteins_relations.protein_source.unique()) if elemento not in list(df_avg_aminoacid_protein_source.protein_source.unique())]
    df_proteins_without_amino_acids_2 = pd.DataFrame(data={'protein_source':proteins_without_aminoacids})
    for idx, row in df_proteins_without_amino_acids_2.iterrows():
        df_avg_aminoacid_protein_source.loc[len(df_avg_aminoacid_protein_source),'protein_source'] = row['protein_source'] 
        for variable in df_avg_aminoacid_protein_source.columns[1:]:
            var_mean = df_avg_aminoacid_protein_source[variable].mean()
            df_avg_aminoacid_protein_source.loc[df_avg_aminoacid_protein_source['protein_source'] == row['protein_source'], variable] = var_mean


    # save nutritional theoretical values per protein source
    df_avg_aminoacid_protein_source = df_avg_aminoacid_protein_source.round(2)
    df_avg_aminoacid_protein_source.to_sql('aminoacids_theoretical_values', engine, if_exists='replace', index=False)
    # aminoacid avg
    df_avg_aminoacid = df_data_group_pivot.describe().transpose().reset_index()
    df_avg_aminoacid = df_avg_aminoacid.round(2)
    df_avg_aminoacid.rename(columns={'':'amino_acid'}, inplace=True)

    # DATA IMPUTATION
    print('Starting amino acid data imputation...')
    #   1 - If the variable value is NA then impute with the amino acid mean value of the same protein source 
    for idx, row in df_data_group_pivot.iterrows():
        for jdx, item in row[row.isna()].items():
            df_data_group_pivot.loc[idx,jdx] = df_avg_aminoacid_protein_source.loc[df_avg_aminoacid_protein_source.protein_source == row['protein_source'], jdx].values[0]

    #   2 - If stills being NA impute with the amino acid mean value
    for idx, row in df_data_group_pivot.iterrows():
        for jdx, item in row[row.isna()].items():
            df_data_group_pivot.loc[idx,jdx] = df_avg_aminoacid.loc[df_avg_aminoacid.amino_acid == jdx, 'mean'].values[0]

    # SAVE CLEAN DATAFRAME TO THE DATABASE
    df_data_group_pivot.to_sql('aminoacid_data_imputation_model_input', engine, if_exists='replace', index=False)

    # NORMALISATION MIN-MAX OVER THE CLEAN DATASET
    print('Starting amino acid data normalisation...')
    df_final_aminoacid_scaled = df_data_group_pivot.copy()

    for column in df_final_aminoacid_scaled.columns[2:]:
        df_final_aminoacid_scaled[column] = (df_final_aminoacid_scaled[column] - df_final_aminoacid_scaled[column].min()) / (df_final_aminoacid_scaled[column].max() - df_final_aminoacid_scaled[column].min())

    df_final_aminoacid_scaled.to_sql('aminoacid_data_imputation_model_input_normalised', engine, if_exists='replace', index=False)

    return

def generateCleanDatasetEnvironmental(engine):
    ###################################################
    #   ENVIRONMENTAL DATASET CLEANING AND TRANSFORMATION
    ###################################################
    # Step 1: Extract raw data from the database
    # Load food sources data
    selectProteinSources = "SELECT\
                            *\
                        FROM\
                            public.protein_source_format_gm psfg"

    selectProteinRelations = "SELECT\
                                *\
                            FROM\
                                dcf_data.food_data_source_relations fdsr;"

    selectEnvironmentalSustainabilityFacts = "SELECT \
                                            *\
                                        from\
                                            public.protein_source_format_gm psfg\
                                        INNER JOIN dcf_data.food_data_source_relations fdsr ON fdsr.food_product_id = psfg.id\
                                        inner join dcf_data.combined_inventory_activities cia on cia.food_data_source_relation_id = fdsr.id \
                                        inner join dcf_data.elca_environment_indicator_facts eeif on cia.id = eeif.activity_id;"

    selectEnvironmentalSustainabilityIndicators = "SELECT DISTINCT eei.indicator_name, eeif.env_indicator_id,eeif.unit FROM dcf_data.elca_environment_indicator_facts eeif INNER JOIN dcf_data.elca_environment_indicators eei ON eei.id = eeif.env_indicator_id;"

    with engine.connect() as connection:
        trans = connection.begin()
        result_proteinSources = pd.read_sql_query(text(selectProteinSources), con=connection)
        result_proteinRelations = pd.read_sql_query(text(selectProteinRelations), con=connection)
        resultEnvironmentalSustainabilityFacts = pd.read_sql_query(text(selectEnvironmentalSustainabilityFacts), con=connection)
        resultEnvironmentalSustainabilityIndicators = pd.read_sql_query(text(selectEnvironmentalSustainabilityIndicators), con=connection)
        trans.commit()

    df_food_sources = pd.DataFrame(result_proteinSources)
    df_food_relations = pd.DataFrame(result_proteinRelations)
    df_food_relations.rename(columns={"id":"relation_id"}, inplace=True)
    df_environmental_sustainability = pd.DataFrame(resultEnvironmentalSustainabilityFacts)
    df_environmental_sustainability_indicators = pd.DataFrame(resultEnvironmentalSustainabilityIndicators)

    # Step 2: Process and transform dataset
    #   Get and save informed environmental sustainability indicators

    #   Merge food sources with domains relations
    df_proteins = df_food_sources[~df_food_sources.protein_source.isna()][['id','protein_source_or_food_product', 'protein_source']]
    df_proteins_relations = df_proteins.merge(df_food_relations, how = 'inner', left_on='id', right_on='food_product_id', suffixes=('', '_remove'))
    df_proteins_relations.drop([i for i in df_proteins_relations.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    #  Merge with environmental sustainability facts
    df_proteins_relations_environmental_sustainability = df_proteins_relations.merge(df_environmental_sustainability, how='inner', left_on='relation_id', right_on='food_data_source_relation_id', suffixes=('', '_remove'))
    df_proteins_relations_environmental_sustainability.drop([i for i in df_proteins_relations_environmental_sustainability.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    # !!!!!!! IF IT WAS STANDARDISED IT, THE MERGE WILL BE MADE JUST USING INDICATOR_ID
    df_proteins_complete = df_proteins_relations_environmental_sustainability.merge(df_environmental_sustainability_indicators, how='inner', left_on=['env_indicator_id','unit'], right_on=['env_indicator_id','unit'], suffixes=('', '_remove'))
    df_proteins_complete.drop([i for i in df_proteins_complete.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns

    # Clean syntax
    df_proteins_complete['amount_or_impact_value'] = df_proteins_complete['amount_or_impact_value'].astype(float)

    # Select environmental indicators that are informed

    df_lookup_environmental_indicators = pd.DataFrame({"env_indicator_name":['Resource use, minerals and metals','Land use','Human toxicity, cancer','Climate change','Acidification','Eutrophication, terrestrial',
                                    'Ecotoxicity, freshwater - inorganics','Human toxicity, cancer - organics','Climate change - Land use and LU change','Climate change - Biogenic',
                                    'Climate change - Fossil','Human toxicity, cancer - inorganics','Ionising radiation','Human toxicity, non-cancer','Eutrophication, freshwater',
                                    'Human toxicity, non-cancer - organics','Photochemical ozone formation','Eutrophication, marine','Resource use, fossils','Ozone depletion',
                                    'Human toxicity, non-cancer - inorganics','Particulate matter','Water use'],
                "unit":['kg Sb eq','Pt','CTUh','kg CO2 eq','mol H+ eq','mol N eq','CTUe','CTUh','kg CO2 eq','kg CO2 eq','kg CO2 eq','CTUh','kBq U-235 eq','CTUh','kg P eq','CTUh','kg NMVOC eq',
                        'kg N eq','MJ','kg CFC11 eq','CTUh','disease inc.','m3 depriv.']})

    # Select only climate change indicator for testing
    df_proteins_complete_selected = df_proteins_complete[(df_proteins_complete.indicator_name == 'Climate change') & (df_proteins_complete.unit == 'kg CO2 eq')]
    df_proteins_complete_selected['indicator_name_unit'] = df_proteins_complete_selected['indicator_name'] + ' (' + df_proteins_complete_selected['unit'] + ')' # merge nutrient name and nutrient unit

    # Multiple reported amounts for the same nutrient and same protein, calculate the mean
    df_proteins_complete_selected_avg = df_proteins_complete_selected.groupby(['protein_source_or_food_product','protein_source','indicator_name_unit']).agg({'amount_or_impact_value':'mean'}).round(2).reset_index()


    # Data is ready to pivot
    df_data_group_pivot = df_proteins_complete_selected_avg.pivot(index=['protein_source_or_food_product','protein_source'], columns='indicator_name_unit', values='amount_or_impact_value').reset_index()
    #
    df_data_group_pivot.columns.name = ''

    # CALCULATES AVG VALUES OF ENVIRONMENTAL INDICATORS PER PROTEIN SOURCE AND AVG ENV INDICATOR VALUES
    # environmental indicators for each protein source
    df_avg_environmental_protein_source = df_data_group_pivot.iloc[:,1:].groupby(['protein_source']).mean().reset_index()
    df_avg_environmental_protein_source = df_avg_environmental_protein_source.round(2)

    # !!! TO IMPROVE: For some reason there are some protein sources that are in the nutrients and amino acids dataset but not in the environment dataset and they are not being captured by the query above. So, I will check for those protein sources and impute their values as well.
    proteins_without_aminoacids = [elemento for elemento in list(df_proteins_relations.protein_source.unique()) if elemento not in list(df_avg_environmental_protein_source.protein_source.unique())]
    df_proteins_without_amino_acids_2 = pd.DataFrame(data={'protein_source':proteins_without_aminoacids})
    for idx, row in df_proteins_without_amino_acids_2.iterrows():
        df_avg_environmental_protein_source.loc[len(df_avg_environmental_protein_source),'protein_source'] = row['protein_source'] 
        for variable in df_avg_environmental_protein_source.columns[1:]:
            var_mean = df_avg_environmental_protein_source[variable].mean()
            df_avg_environmental_protein_source.loc[df_avg_environmental_protein_source['protein_source'] == row['protein_source'], variable] = var_mean

    # save environmental theoretical values per protein source
    df_avg_environmental_protein_source.to_sql('environmental_theoretical_values', engine, if_exists='replace', index=False)

    # environmental indicator avg
    df_avg_environmental = df_data_group_pivot.describe().transpose().reset_index()
    df_avg_environmental = df_avg_environmental.round(2)
    df_avg_environmental.rename(columns={'':'environmental_indicator'}, inplace=True)

    # DATA IMPUTATION
    print('Starting environmental data imputation...')

    #   1 - If the variable value is NA then impute with the environmental mean value of the same protein source
    for idx, row in df_data_group_pivot.iterrows():
        print(row['protein_source_or_food_product'])
        for jdx, item in row[row.isna()].items():
            if df_avg_environmental_protein_source.loc[(df_avg_environmental_protein_source.protein_source == row['protein_source']), jdx].empty:
                df_data_group_pivot.loc[idx,jdx] = df_avg_environmental_protein_source.loc[(df_avg_environmental_protein_source.protein_source == row['protein_source']), jdx].values[0]
                break
            else:
                df_data_group_pivot.loc[idx,jdx] = df_avg_environmental_protein_source.loc[(df_avg_environmental_protein_source.protein_source == row['protein_source']), jdx].values[0]

    #   2 - If stills being NA impute with the environmental mean value
    for idx, row in df_data_group_pivot.iterrows():
        for jdx, item in row[row.isna()].items():
            df_data_group_pivot.loc[idx,jdx] = df_avg_environmental_protein_source.loc[df_avg_environmental_protein_source.protein_source == row['protein_source'], jdx].values[0]

    # SAVE CLEAN DATAFRAME TO THE DATABASE
    df_data_group_pivot.to_sql('environmental_data_imputation_model_input', engine, if_exists='replace', index=False)


    # NORMALISATION MIN-MAX OVER THE CLEAN DATASET
    print('Starting environmental data normalisation...')
    df_final_environmental_scaled = df_data_group_pivot.copy()

    for column in df_final_environmental_scaled.columns[2:len(df_final_environmental_scaled.columns)-2]:
        df_final_environmental_scaled[column] = (df_final_environmental_scaled[column] - df_final_environmental_scaled[column].min()) / (df_final_environmental_scaled[column].max() - df_final_environmental_scaled[column].min())

    df_final_environmental_scaled.to_sql('environmental_data_imputation_model_input_normalised', engine, if_exists='replace', index=False)

    return


def AltProtRecommendation(data: data_models.userProteinInput, engine):

    # 0 - Define pipelines for the model. The same pipelines will be applied to the input data and to the data of the model (already scaled during data treatment)
    # Base pipeline: scaled (already scaled during data treatment) + model
    pipe_knn_custom = Pipeline(steps=[
        # ("scaler", StandardScaler()),
        ("model", KNN(k=10))
    ])

    # Pipeline with PCA + kNN custom
    pipe_pca75_knn_custom = Pipeline(steps=[
        # ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.75, svd_solver='full',random_state=42)),
        ("model", KNN(k=10))
    ])

    # Pipeline with PCA + kNN custom
    pipe_pca85_knn_custom = Pipeline(steps=[
        # ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.85, svd_solver='full',random_state=42)),
        ("model", KNN(k=10))
    ])

    # Pipeline with PCA + kNN custom
    pipe_pca90_knn_custom = Pipeline(steps=[
        # ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.90, svd_solver='full',random_state=42)),
        ("model", KNN(k=10))
    ])

    # Define relevant variables for the model (nutrients and amino acids)
    relevant_variables = ["protein_source", "Fat (g)", "Glutamic acid (Glu/E) (g/100 g Protein)", "Valine (Val/V) (g/100 g Protein)", "Phenylalanine (Phe/F) (g/100 g Protein)",
    "Phosphorus (mg)", "Moisture (g)", "Carbohydrate (g)", "Polyunsaturated fatty acids (PUFA) (g)", "Chromium (ug)", "Proline (Pro/P) (g/100 g Protein)", "Total sugar (g)",
    "Starch (g)", "Protein (g)","Climate change (kg CO2 eq)"]

    # Check if the protein ID exists in the database
    selectProteinId = "SELECT \
                                * \
                            FROM \
	                            dcf_data.food_sources fs2 \
                            WHERE \
	                            fs2.id = '{}'".format(data.proteinId)

    with engine.connect() as connection:
        trans = connection.begin()
        result_proteinId = pd.read_sql_query(text(selectProteinId), con=connection)
        trans.commit()

    try:
        result_proteinId.values[0]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item ID does not exist in database")

    # 1 - The protein ID (protein name) is received and query the characteristics of the protein food

    selectProteinSourceAndCategory = "SELECT \
                                psfg.protein_source, \
                                psfg.category \
                            FROM \
	                            dcf_data.food_sources fs2 \
                            LEFT JOIN protein_source_format_gm psfg ON fs2.id = psfg.id \
                            WHERE \
	                            fs2.id = '{}'".format(data.proteinId)

    selectProteinNutrients = "SELECT\
                                psfg.id,\
                                ndim.*\
                            FROM\
                                public.nutrients_data_imputation_model_input_normalised ndim\
                            INNER JOIN protein_source_format_gm psfg ON ndim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                            WHERE\
                                psfg.id = '{}';".format(data.proteinId)

    selectProteinAminoAcids = "SELECT\
                                psfg.id,\
                                adim.*\
                            FROM\
                                public.aminoacid_data_imputation_model_input_normalised adim\
                            INNER JOIN protein_source_format_gm psfg ON adim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                            WHERE\
                                psfg.id = '{}';".format(data.proteinId)

    selectProteinEnvironmental = "SELECT\
                                psfg.id,\
                                edim.*\
                            FROM\
                                public.environmental_data_imputation_model_input_normalised edim\
                            INNER JOIN protein_source_format_gm psfg ON edim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                            WHERE\
                                psfg.id = '{}';".format(data.proteinId)

    with engine.connect() as connection:
        trans = connection.begin()
        result_source_type = pd.read_sql_query(text(selectProteinSourceAndCategory), con=connection)
        result_nutrients = pd.read_sql_query(text(selectProteinNutrients), con=connection)
        result_aminoacids = pd.read_sql_query(text(selectProteinAminoAcids), con=connection)
        result_environmental = pd.read_sql_query(text(selectProteinEnvironmental), con=connection)
        trans.commit()

    source_type = result_source_type.values[0][0]
    category = result_source_type.values[0][1]

    if category is None or category == 'unknown':
        category = 'ingredient'

    df_protein_source_nutrients = pd.DataFrame(result_nutrients)
    # If there are no normalised nutrient data for the (protein) item, get the raw data and process it
    if df_protein_source_nutrients.empty:
        df_protein_source_nutrients = getInputProteinNutrientData(data.proteinId, engine)

    df_protein_source_aminoacids = pd.DataFrame(result_aminoacids)
    # If there are no normalised amino acid data for the (protein) item, get the raw data and process it
    if df_protein_source_aminoacids.empty:
        df_protein_source_aminoacids = getInputProteinAminoAcidData(data.proteinId, engine)
    
    df_protein_source_environmental = pd.DataFrame(result_environmental)
    # If there are no normalised environmental data for the (protein) item, get the raw data and process it
    if df_protein_source_environmental.empty:
        df_protein_source_environmental = getInputProteinEnvironmentalData(data.proteinId, engine)

    # Merge nutrient and amino acid data for the input protein
    df_input_protein = df_protein_source_nutrients.merge(df_protein_source_aminoacids, how='inner', left_on=['id'], right_on=['id'], suffixes=('', '_remove'))
    df_input_protein.drop([i for i in df_input_protein.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns
    # Merge nutrient, amino acid and environmental data for the input protein
    df_input_protein = df_input_protein.merge(df_protein_source_environmental, how='inner', left_on=['id'], right_on=['id'], suffixes=('', '_remove'))
    df_input_protein.drop([i for i in df_input_protein.columns if 'remove' in i], axis=1, inplace=True) # drop duplicated columns
    # 2 - Select nutrient, amino acid, environmental data already treated for the model. Exclude traditional protein sources and the same category as the input protein
    with engine.connect() as connection:
        trans = connection.begin()
        # 2.1 - Select nutrient data for the model
        result_nutrients_data_model = pd.read_sql_query(text("SELECT\
                                                                psfg.id,\
                                                                ndim.*\
                                                            FROM\
                                                                public.nutrients_data_imputation_model_input_normalised ndim\
                                                            INNER JOIN protein_source_format_gm psfg ON ndim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                                            INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                                            WHERE\
                                                                pst.protein_source != 'Traditional' \
                                                                AND psfg.category = '{}' \
                                                                AND psfg.id != '{}';".format(category, data.proteinId)) , con=connection)
        # 2.2 - Select nutrients, amino acid, environmental data for the model
        result_nutrients_aminoacid_environmental_data_model = pd.read_sql_query(text("WITH aminoacids AS (\
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
                                        LEFT JOIN public.aminoacid_data_imputation_model_input_normalised adimin ON psfg.protein_source_or_food_product = adimin.protein_source_or_food_product \
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
                                        LEFT JOIN public.environmental_data_imputation_model_input_normalised edimin ON psfg.protein_source_or_food_product = edimin.protein_source_or_food_product \
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
                                        INNER JOIN public.nutrients_data_imputation_model_input_normalised ndim ON ndim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        INNER JOIN aminoacids aa ON aa.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        INNER JOIN environmental_impact ei ON ei.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        WHERE\
                                            pst.source_type != 'Traditional';"), con=connection)

        trans.commit()

    df_protein_data_model = pd.DataFrame(result_nutrients_aminoacid_environmental_data_model)
    df_protein_data_model.drop_duplicates(inplace=True)
    #df_protein_data_model.fillna(0, inplace=True)

    # 3 - Prepare data for k-NN model
    X_model = df_protein_data_model[relevant_variables].select_dtypes(include='float').values
    X_model = X_model.astype('float32')
    y_model = df_protein_data_model.id.values

    X_predict = df_input_protein[relevant_variables].select_dtypes(include='float').values
    X_predict = X_predict.astype('float32')
    y_predict = df_input_protein.id.values

    # 4 - Create PCA + k-NN model and fit with the data of the model (already scaled during data treatment) 
    pipe_pca75_knn_custom.fit(X_model, y_model)
            # clf = KNN(k=10)
            # clf.fit(X_model, y_model)
    # 5 - Predict alternative proteins
    listRecommendations = pipe_pca75_knn_custom.predict(X_predict)

    listAltProt = data_models.ListAlernativeProtein()
    for recomm in listRecommendations:
        for protein_id in recomm[0]:
            with engine.connect() as connection:
                trans = connection.begin()
                protein_name = pd.read_sql_query(text("SELECT\
                                                                        psfg.protein_source_or_food_product\
                                                                    FROM\
                                                                        protein_source_format_gm psfg\
                                                                    WHERE\
                                                                        psfg.id = '{}';".format(protein_id)) , con=connection)

                trans.commit()
            print(protein_id)
            print(protein_name.values[0])
            listAltProt.alternativeProteins.append(protein_id)
    

    #######################
    # 6 - INSERT INTO DATABASE THE ALGORITHM RECOMMENDATIONS FOR PROTEIN SPECIFIED
    #######################
    query_insert_requests = "INSERT INTO public.protein_screening_algorithm_requests \
    (protein_source_id, recommended_alternative_protein_source_ids) \
    VALUES (:protein_id, :alt_proteins);"

    data.proteinId, listAltProt.alternativeProteins


    with engine.connect() as connection:
        trans = connection.begin()
        connection.execute(text(query_insert_requests), { "protein_id": data.proteinId, "alt_proteins": listAltProt.alternativeProteins })
        trans.commit()
    return listAltProt

