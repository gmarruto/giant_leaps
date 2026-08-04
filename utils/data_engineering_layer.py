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
    proteins_without_environmental = [elemento for elemento in list(df_proteins_relations.protein_source.unique()) if elemento not in list(df_avg_environmental_protein_source.protein_source.unique())]
    df_proteins_without_environmental_2 = pd.DataFrame(data={'protein_source':proteins_without_environmental})
    for idx, row in df_proteins_without_environmental_2.iterrows():
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

    for column in df_final_environmental_scaled.columns[2:len(df_final_environmental_scaled.columns)]:
        df_final_environmental_scaled[column] = (df_final_environmental_scaled[column] - df_final_environmental_scaled[column].min()) / (df_final_environmental_scaled[column].max() - df_final_environmental_scaled[column].min())

    df_final_environmental_scaled.to_sql('environmental_data_imputation_model_input_normalised', engine, if_exists='replace', index=False)

    return
