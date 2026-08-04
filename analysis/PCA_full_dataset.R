
library(readxl)
library(writexl)
library(GWalkR)
if (!require(tidyverse)) install.packages("tidyverse")
library(tidyverse)
library(ggplot2)
if (!require(ggfortify)) install.packages("ggfortify")
library(gridExtra)
library(ggfortify)
library(psych)
library(rjson)
library(RPostgreSQL)
library(fields)
library(viridis)

source("EDA_utils.R")

# ====================
#   GLOBAL VARIABLES
# ====================
db_connections <- "../others/db_credentials.json";


# ========================
#      GET DATA
# ========================


# 03/02/2026 - Analysis after data cleaning, transformation, standardisation and imputation

# Get Nutrients Data Imputation Model Input
postgres_conn <- connect_PostgreSQL(db_connections, db_name = "no_ssh_giantleaps")

query_theoretical_nutrients_data <- "SELECT 
                                    	*
                                    FROM
                                    	nutrients_theoretical_values ntv;"

query_nutrients_data <- "SELECT
                        	*
                        FROM
                        	nutrients_data_imputation_model_input_normalised ndimin;"


query_theoretical_nutrients_data_res <- dbGetQuery(postgres_conn, query_theoretical_nutrients_data)
df_theoretical_nutrients_data <- as.data.frame(query_theoretical_nutrients_data_res)

query_nutrients_data_res <- dbGetQuery(postgres_conn, query_nutrients_data)
df_nutrients_data <- as.data.frame(query_nutrients_data_res)

# Get Amino Acid Data Imputation Model Input
query_theoretical_aminoacid_data <- "SELECT
                                    	*
                                    FROM
                                    	aminoacid_theoretical_values atv;"

query_aminoacid_data <- "SELECT
                        	*
                        FROM
                        	aminoacid_data_imputation_model_input_normalised adimin;"

query_theoretical_aminoacid_data_res <- dbGetQuery(postgres_conn, query_theoretical_aminoacid_data)
df_theoretical_aminoacid_data <- as.data.frame(query_theoretical_aminoacid_data_res)

query_aminoacid_data_res <- dbGetQuery(postgres_conn, query_aminoacid_data)
df_aminoacid_data <- as.data.frame(query_aminoacid_data_res)

# Get Environmental Impact Data Imputation Model Input
query_theoretical_environmental <- "SELECT
                                    	*
                                    FROM
                                    	environmental_theoretical_values etv;"

query_environmental_data <- "SELECT
                        	*
                        FROM
                        	environmental_data_imputation_model_input_normalised edimin;"

query_theoretical_environmental_data_res <- dbGetQuery(postgres_conn, query_theoretical_environmental)
df_theoretical_environmental_data <- as.data.frame(query_theoretical_environmental_data_res)

query_environmental_data_res <- dbGetQuery(postgres_conn, query_environmental_data)
df_envrionmental_data <- as.data.frame(query_environmental_data_res)


# Get Nutritional and Amino Acid Imputation Model Input

query_nutrients_aminoacids_data_model <- "WITH aminoacids AS (\
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
                                        )\
                                        SELECT\
                                            psfg.id,\
                                            psfg.category,\
                                            ndim.*,\
                                            aa.*\
                                        FROM\
                                            protein_source_format_gm psfg\
                                        INNER JOIN protein_source_type pst ON pst.protein_source = psfg.protein_source\
                                        INNER JOIN public.nutrients_data_imputation_model_input_normalised ndim ON ndim.protein_source_or_food_product = psfg.protein_source_or_food_product\
                                        INNER JOIN aminoacids aa ON aa.protein_source_or_food_product = psfg.protein_source_or_food_product;"

query_nutrients_aminoacids_data_model <- dbGetQuery(postgres_conn, query_nutrients_aminoacids_data_model)
df_nutrients_aminoacids_data_model <- as.data.frame(query_nutrients_aminoacids_data_model)


# Get Nutritional, Amino Acid and Environmental Data Imputation Model Input

query_nutrients_aminoacids_environmental_data_model <- "WITH aminoacids AS (\
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
                                          LEFT JOIN public.environmental_data_imputation_model_input_normalised edimin ON psfg.protein_source_or_food_product = edimin.protein_source_or_food_product\
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
                                        INNER JOIN aminoacids aa ON aa.protein_source_or_food_product = psfg.protein_source_or_food_product
                                        INNER JOIN environmental_impact ei on ei.protein_source_or_food_product = psfg.protein_source_or_food_product;"

query_nutrients_aminoacids_environmental_data_model <- dbGetQuery(postgres_conn, query_nutrients_aminoacids_environmental_data_model)
df_nutrients_aminoacids_environmental_data_model <- as.data.frame(query_nutrients_aminoacids_environmental_data_model)

# ==========================================



# =============================================
#   PCA WITH NUTRITIONAL THEORETICAL VALUES
#     1- Data imputation for theoretical value 
#     2- Principal Component Analysis. 
#         Standardise data (Center and Scale)
# =============================================

# 1- Data imputation for theoretical value.

for (variable in colnames(df_theoretical_nutrients_data)[c(4:length(df_theoretical_nutrients_data))]) {
  #print(variable)
  for(row in 1:nrow(df_theoretical_nutrients_data)){
    if (is.na(df_theoretical_nutrients_data[row, variable])){
      #print(mean(na.exclude(df_theoretical_nutrients_data[df_theoretical_nutrients_data$protein_source ==  df_theoretical_nutrients_data$protein_source[row], variable])))
      if (!is.na(mean(na.exclude(df_theoretical_nutrients_data[df_theoretical_nutrients_data$protein_source ==  df_theoretical_nutrients_data$protein_source[row], variable])))){
        df_theoretical_nutrients_data[row, variable] = sapply(df_theoretical_nutrients_data[df_theoretical_nutrients_data$protein_source ==  df_theoretical_nutrients_data$protein_source[row],][variable], mean, na.rm = TRUE)
        #print(row)
        }
      else{
        df_theoretical_nutrients_data[row, variable] = sapply(df_theoretical_nutrients_data[variable], mean, na.rm = TRUE)
      }
    }
  }
  
}

# 2- Principal Component Analysis.

df_theoretical_nutrients_data_PCA = within(df_theoretical_nutrients_data, rm('Moisture Level Num'))
colnames(df_theoretical_nutrients_data_PCA)[2] <- "Moisture_Level"
df_theoretical_nutrients_data_PCA$Moisture_Level <- factor(df_theoretical_nutrients_data_PCA$Moisture_Level, levels=c('Very High', 'High', 'Intermediate', 'Low', 'Very Low'))
PCA_analysis(df_theoretical_nutrients_data_PCA, "Moisture_Level", "protein_source", "PCA theoretical nutrients", file_dir = NULL)

#     >>> Exclude variables with a variance close to zero
numeric_variables <- unlist(lapply(df_theoretical_nutrients_data_PCA, is.numeric), use.names = FALSE)
variables_var <- sapply(df_theoretical_nutrients_data_PCA[,numeric_variables], var)
selected_variables <- names(variables_var[!variables_var < 0.001])
df_theoretical_nutrients_data_PCA <- df_theoretical_nutrients_data_PCA[c("protein_source", "Moisture_Level", selected_variables)]

#     >>> Avoid high correlated variables
corr_matrix_nutrients_PCA <- cor(df_theoretical_nutrients_data_PCA[,selected_variables], method="pearson")
high_corr_paired_vars <- which(corr_matrix_nutrients_PCA > 0.85 | corr_matrix_nutrients_PCA < -0.85, arr.ind = TRUE) # >>>> There is no high correlated pair of variables despite the diagonal 

for (row in 1:(length(high_corr_paired_vars)/2)){
  if (high_corr_paired_vars[row,1] != high_corr_paired_vars[row,2]){
    print(paste0(paste0(colnames(corr_matrix_nutrients_PCA)[high_corr_paired_vars[row,1]],' - '),colnames(corr_matrix_nutrients_PCA)[high_corr_paired_vars[row,2]]))
  }
}

selected_variables_1 <- selected_variables[!selected_variables %in% c("Linoleic acid (18:3, w6) (g)", "Polyunsaturated fatty acids (PUFA) (g)", "Ash (g)", "Energy (kcal)",
                                                                            "Monounsaturated fatty acids (MUFA) (g)", "Phosphorus (mg)", "Sucrose (g)")] 
df_theoretical_nutrients_data_PCA_sel1 <- df_theoretical_nutrients_data_PCA[c("protein_source", "Moisture_Level", selected_variables_1)]


df_theoretical_nutrients_data_PCA_sel1$Moisture_Level <- factor(df_theoretical_nutrients_data_PCA_sel1$Moisture_Level, levels=c('Very High', 'High', 'Intermediate', 'Low', 'Very Low'))
PCA_analysis(df_theoretical_nutrients_data_PCA_sel1, "Moisture_Level", "protein_source", "PCA theoretical nutrients (excluded low variance)", file_dir = NULL)

#     >>> Compute contribution of each variable to the new space of PCs and select the 10 most relevant

results_variable_contributions <- pca_variable_contributions(df_theoretical_nutrients_data_PCA_sel1,
                                                             method_cor = "pearson",
                                                             impute_median = TRUE,
                                                             var_expl_acum = 0.90,   # usa los PCs necesarios hasta el 90% de varianza acumulada
                                                             top_n = 15)

df_theoretical_nutrients_data_PCA_sel2 <- df_theoretical_nutrients_data_PCA_sel1[c("protein_source", "Moisture_Level", names(results_variable_contributions$top_vars))]

PCA_analysis(df_theoretical_nutrients_data_PCA_sel2, "Moisture_Level", "protein_source", "PCA theoretical nutrients (PCA high contribution variables)", file_dir = NULL)


#     >>> Include Protein (g) variable in the selection
df_theoretical_nutrients_data_PCA_sel3 <- df_theoretical_nutrients_data_PCA_sel1[c("protein_source", "Moisture_Level", "Protein (g)", names(results_variable_contributions$top_vars))]

PCA_analysis(df_theoretical_nutrients_data_PCA_sel3, "Moisture_Level", "protein_source", "PCA theoretical nutrients (PCA high contribution variables and Protein)", file_dir = NULL)




# =============================================
#   PCA WITH AMINO ACIDS THEORETICAL VALUES
#     1- Data imputation for theoretical value 
#     2- Principal Component Analysis. 
#         Standardise data (Center and Scale)
# =============================================

# 1- Data imputation for theoretical value.

for (variable in colnames(df_theoretical_aminoacid_data)[c(2:length(df_theoretical_aminoacid_data))]) {
  #print(variable)
  for(row in 1:nrow(df_theoretical_aminoacid_data)){
    if (is.na(df_theoretical_aminoacid_data[row, variable])){
      if (!is.na(mean(na.exclude(df_theoretical_aminoacid_data[df_theoretical_aminoacid_data$protein_source ==  df_theoretical_aminoacid_data$protein_source[row], variable])))){
        df_theoretical_aminoacid_data[row, variable] = sapply(df_theoretical_aminoacid_data[df_theoretical_aminoacid_data$protein_source ==  df_theoretical_aminoacid_data$protein_source[row],][variable], mean, na.rm = TRUE)
      }
      else{
        df_theoretical_aminoacid_data[row, variable] = sapply(df_theoretical_aminoacid_data[variable], mean, na.rm = TRUE)
      }
    }
  }
  
}

# 2- Principal Component Analysis.

#df_theoretical_aminoacid_data_PCA = within(df_theoretical_aminoacid_data, rm('Moisture Level Num'))
#colnames(df_theoretical_aminoacid_data_PCA)[2] <- "Moisture_Level"
#df_theoretical_aminoacid_data_PCA$Moisture_Level <- factor(df_theoretical_aminoacid_data_PCA$Moisture_Level, levels=c('Very High', 'High', 'Intermediate', 'Low', 'Very Low'))
df_theoretical_aminoacid_data_PCA <- df_theoretical_aminoacid_data
PCA_analysis(df_theoretical_aminoacid_data_PCA, "protein_source", NA, "PCA theoretical amino acids", file_dir = NULL)

#     >>> Exclude variables with a variance close to zero
numeric_variables <- unlist(lapply(df_theoretical_aminoacid_data_PCA, is.numeric), use.names = FALSE)
variables_var <- sapply(df_theoretical_aminoacid_data_PCA[,numeric_variables], var)
selected_variables <- names(variables_var[!variables_var < 0.001])
df_theoretical_aminoacid_data_PCA <- df_theoretical_aminoacid_data_PCA[c("protein_source", selected_variables)]

#     >>> Avoid high correlated variables
corr_matrix_aminoacids_PCA <- cor(df_theoretical_aminoacid_data_PCA[,selected_variables], method="pearson")
high_corr_paired_aminoacid_vars <- which(corr_matrix_aminoacids_PCA > 0.90 | corr_matrix_aminoacids_PCA < -0.90, arr.ind = TRUE) # >>>> There is no high correlated pair of variables despite the diagonal 

for (row in 1:(length(high_corr_paired_aminoacid_vars)/2)){
  if (high_corr_paired_aminoacid_vars[row,1] != high_corr_paired_aminoacid_vars[row,2]){
    print(paste0(paste0(colnames(corr_matrix_aminoacids_PCA)[high_corr_paired_aminoacid_vars[row,1]],' - '),colnames(corr_matrix_aminoacids_PCA)[high_corr_paired_aminoacid_vars[row,2]]))
    print(corr_matrix_aminoacids_PCA[high_corr_paired_aminoacid_vars[row,1], high_corr_paired_aminoacid_vars[row,2]])
    }
}

selected_variables_1 <- selected_variables[!selected_variables %in% c("Arginine  (Arg/R) (g/100g protein)",
                                                                      "Aspartic Acid (Asp/D) (g/100g protein)" ,
                                                                      "Glutamic acid (Glu/E) (g/100g protein)",
                                                                      "Glycine (Gly/G) (g/100g protein)",
                                                                      "Histidine  (His/H) (g/100g protein)",
                                                                      "Isoleucine (Ile/I) (g/100g protein)",
                                                                      "Leucine (Leu/L) (g/100g protein)",
                                                                      "Lysine (Lys/K) (g/100g protein)",
                                                                      "Methionine (Met/M) (g/100g protein)",
                                                                      "Phenylalanine (Phe/F) (g/100g protein)",
                                                                      "Proline (Pro/P) (g/100g protein)",
                                                                      "Serine (Ser/S) (g/100g protein)",
                                                                      "Threonine (Thr/T) (g/100g protein)",
                                                                      "Valine (Val/V) (g/100g protein)")]
df_theoretical_aminoacid_data_PCA_sel1 <- df_theoretical_aminoacid_data_PCA[c("protein_source", selected_variables_1)]

PCA_analysis(df_theoretical_aminoacid_data_PCA_sel1, "protein_source", NA, "PCA theoretical amino acids (excluded low variance)", file_dir = NULL)

#     >>> Compute contribution of each variable to the new space of PCs and select the 10 most relevant

results_variable_contributions <- pca_variable_contributions(df_theoretical_aminoacid_data_PCA,
                                                             method_cor = "pearson",
                                                             impute_median = TRUE,
                                                             var_expl_acum = 0.90,   # usa los PCs necesarios hasta el 90% de varianza acumulada
                                                             top_n = 1)

df_theoretical_aminoacid_data_PCA_sel2 <- df_theoretical_aminoacid_data_PCA[c("protein_source", names(results_variable_contributions$top_vars), "Tryptophan (Trp/W) (g/100g protein)")]

PCA_analysis(df_theoretical_aminoacid_data_PCA_sel2, "protein_source", NA,"PCA theoretical amino acids (PCA high contribution variables)", file_dir = NULL)



# =============================================
#   PCA WITH NUTRIENTS + AMINO ACIDS THEORETICAL VALUES
#     2- Principal Component Analysis. 
#         Standardise data (Center and Scale)
# =============================================


# 2- Principal Component Analysis.

df_theoretical_nutrients_aminoacid_data_PCA <- merge(df_theoretical_nutrients_data, df_theoretical_aminoacid_data)
df_theoretical_nutrients_aminoacid_data_PCA = within(df_theoretical_nutrients_aminoacid_data_PCA, rm('Moisture Level Num'))
colnames(df_theoretical_nutrients_aminoacid_data_PCA)[2] <- "Moisture_Level"
df_theoretical_nutrients_aminoacid_data_PCA$Moisture_Level <- factor(df_theoretical_nutrients_aminoacid_data_PCA$Moisture_Level, levels=c('Very High', 'High', 'Intermediate', 'Low', 'Very Low'))
PCA_analysis(df_theoretical_nutrients_aminoacid_data_PCA, "Moisture_Level", "protein_source", "PCA theoretical nutrients and amino acids", file_dir = NULL)

#     >>> Exclude variables with a variance close to zero
numeric_variables <- unlist(lapply(df_theoretical_nutrients_aminoacid_data_PCA, is.numeric), use.names = FALSE)
variables_var <- sapply(df_theoretical_nutrients_aminoacid_data_PCA[,numeric_variables], var)
selected_variables <- names(variables_var[!variables_var < 0.001])
df_theoretical_nutrients_aminoacid_data_PCA <- df_theoretical_nutrients_aminoacid_data_PCA[c("protein_source", "Moisture_Level",selected_variables)]

#     >>> Avoid high correlated variables
corr_matrix_nutrients_aminoacids_PCA <- cor(df_theoretical_nutrients_aminoacid_data_PCA[,selected_variables], method="pearson")
high_corr_paired_nutrients_aminoacid_vars <- which(corr_matrix_nutrients_aminoacids_PCA > 0.85 | corr_matrix_nutrients_aminoacids_PCA < -0.85, arr.ind = TRUE) # >>>> There is no high correlated pair of variables despite the diagonal 

for (row in 1:(length(high_corr_paired_nutrients_aminoacid_vars)/2)){
  if (high_corr_paired_nutrients_aminoacid_vars[row,1] != high_corr_paired_nutrients_aminoacid_vars[row,2]){
    print(paste0(paste0(colnames(corr_matrix_nutrients_aminoacids_PCA)[high_corr_paired_nutrients_aminoacid_vars[row,1]],' - '),colnames(corr_matrix_nutrients_aminoacids_PCA)[high_corr_paired_nutrients_aminoacid_vars[row,2]]))
    print(corr_matrix_nutrients_aminoacids_PCA[high_corr_paired_nutrients_aminoacid_vars[row,1], high_corr_paired_nutrients_aminoacid_vars[row,2]])
  }
}

selected_variables_1 <- selected_variables[!selected_variables %in% c("Arginine  (Arg/R) (g/100g protein)",
                                                                      "Aspartic Acid (Asp/D) (g/100g protein)" ,
                                                                      "Glutamic acid (Glu/E) (g/100g protein)",
                                                                      "Glycine (Gly/G) (g/100g protein)",
                                                                      "Histidine  (His/H) (g/100g protein)",
                                                                      "Isoleucine (Ile/I) (g/100g protein)",
                                                                      "Leucine (Leu/L) (g/100g protein)",
                                                                      "Lysine (Lys/K) (g/100g protein)",
                                                                      "Methionine (Met/M) (g/100g protein)",
                                                                      "Phenylalanine (Phe/F) (g/100g protein)",
                                                                      "Proline (Pro/P) (g/100g protein)",
                                                                      "Serine (Ser/S) (g/100g protein)",
                                                                      "Threonine (Thr/T) (g/100g protein)",
                                                                      "Valine (Val/V) (g/100g protein)", "Linoleic acid (18:3, w6) (g)", 
                                                                      "Polyunsaturated fatty acids (PUFA) (g)", "Ash (g)", "Energy (kcal)",
                                                                      "Monounsaturated fatty acids (MUFA) (g)", "Phosphorus (mg)", "Sucrose (g)")]

df_theoretical_nutrients_aminoacid_data_PCA_sel1 <- df_theoretical_nutrients_aminoacid_data_PCA[c("protein_source", "Moisture_Level", selected_variables_1)]

PCA_analysis(df_theoretical_nutrients_aminoacid_data_PCA_sel1, "Moisture_Level", "protein_source", "PCA theoretical nutrients and amino acids (excluded low variance)", file_dir = NULL)

#     >>> Compute contribution of each variable to the new space of PCs and select the 10 most relevant

results_variable_contributions <- pca_variable_contributions(df_theoretical_nutrients_aminoacid_data_PCA_sel1,
                                                             method_cor = "pearson",
                                                             impute_median = TRUE,
                                                             var_expl_acum = 0.90,   # usa los PCs necesarios hasta el 90% de varianza acumulada
                                                             top_n = 12)

df_theoretical_nutrients_aminoacid_data_PCA_sel2 <- df_theoretical_nutrients_aminoacid_data_PCA_sel1[c("protein_source", "Moisture_Level", names(results_variable_contributions$top_vars), "Protein (g)")]
df_theoretical_nutrients_aminoacid_data_PCA_sel2 <- within(df_theoretical_nutrients_aminoacid_data_PCA_sel2,rm("Fructose (g)", "Tyrosine (Tyr/Y) (g/100g protein)"))


PCA_analysis(df_theoretical_nutrients_aminoacid_data_PCA_sel2, "Moisture_Level", "protein_source", "PCA theoretical nutrients and amino acids (PCA high contribution variables and Protein)", file_dir = NULL)



# ==============================================
#   PCA WITH NUTRIENTS + AMINO ACIDS + ENVIRONMENTAL THEORETICAL VALUES
#     2- Principal Component Analysis. 
#         Standardise data (Center and Scale)
# ==============================================

for (variable in colnames(df_theoretical_environmental_data)[c(2:length(df_theoretical_environmental_data))]) {
  #print(variable)
  for(row in 1:nrow(df_theoretical_environmental_data)){
    if (is.na(df_theoretical_environmental_data[row, variable])){
      if (!is.na(mean(na.exclude(df_theoretical_environmental_data[df_theoretical_environmental_data$protein_source ==  df_theoretical_environmental_data$protein_source[row], variable])))){
        df_theoretical_environmental_data[row, variable] = sapply(df_theoretical_environmental_data[df_theoretical_environmental_data$protein_source ==  df_theoretical_environmental_data$protein_source[row],][variable], mean, na.rm = TRUE)
      }
      else{
        df_theoretical_environmental_data[row, variable] = sapply(df_theoretical_environmental_data[variable], mean, na.rm = TRUE)
      }
    }
  }
  
}

# 2- Principal Component Analysis.

df_theoretical_nutrients_aminoacid_data_PCA <- merge(df_theoretical_nutrients_data, df_theoretical_aminoacid_data)
df_theoretical_nutrients_aminoacid_environmental_data_PCA <- merge(df_theoretical_nutrients_aminoacid_data_PCA, df_theoretical_environmental_data)
df_theoretical_nutrients_aminoacid_environmental_data_PCA = within(df_theoretical_nutrients_aminoacid_environmental_data_PCA, rm('Moisture Level Num'))
colnames(df_theoretical_nutrients_aminoacid_environmental_data_PCA)[2] <- "Moisture_Level"
df_theoretical_nutrients_aminoacid_environmental_data_PCA$Moisture_Level <- factor(df_theoretical_nutrients_aminoacid_environmental_data_PCA$Moisture_Level, levels=c('Very High', 'High', 'Intermediate', 'Low', 'Very Low'))
PCA_analysis(df_theoretical_nutrients_aminoacid_environmental_data_PCA, "Moisture_Level", "protein_source", "PCA theoretical nutrients, amino acids and environmental impact", file_dir = NULL)

#     >>> Exclude variables with a variance close to zero
numeric_variables <- unlist(lapply(df_theoretical_nutrients_aminoacid_environmental_data_PCA, is.numeric), use.names = FALSE)
variables_var <- sapply(df_theoretical_nutrients_aminoacid_environmental_data_PCA[,numeric_variables], var)
selected_variables <- names(variables_var[!variables_var < 0.001])
df_theoretical_nutrients_aminoacid_environmental_data_PCA <- df_theoretical_nutrients_aminoacid_environmental_data_PCA[c("protein_source", "Moisture_Level",selected_variables)]

#     >>> Avoid high correlated variables
corr_matrix_nutrients_aminoacids_environmental_PCA <- cor(df_theoretical_nutrients_aminoacid_environmental_data_PCA[,selected_variables], method="pearson")
high_corr_paired_nutrients_aminoacid_environmental_vars <- which(corr_matrix_nutrients_aminoacids_environmental_PCA > 0.85 | corr_matrix_nutrients_aminoacids_environmental_PCA < -0.85, arr.ind = TRUE) # >>>> There is no high correlated pair of variables despite the diagonal 

for (row in 1:(length(high_corr_paired_nutrients_aminoacid_environmental_vars)/2)){
  if (high_corr_paired_nutrients_aminoacid_environmental_vars[row,1] != high_corr_paired_nutrients_aminoacid_environmental_vars[row,2]){
    print(paste0(paste0(colnames(corr_matrix_nutrients_aminoacids_environmental_PCA)[high_corr_paired_nutrients_aminoacid_environmental_vars[row,1]],' - '),colnames(corr_matrix_nutrients_aminoacids_environmental_PCA)[high_corr_paired_nutrients_aminoacid_environmental_vars[row,2]]))
    print(corr_matrix_nutrients_aminoacids_environmental_PCA[high_corr_paired_nutrients_aminoacid_environmental_vars[row,1], high_corr_paired_nutrients_aminoacid_environmental_vars[row,2]])
  }
}

selected_variables_1 <- selected_variables[!selected_variables %in% c("Arginine  (Arg/R) (g/100g protein)",
                                                                      "Aspartic Acid (Asp/D) (g/100g protein)" ,
                                                                      "Glutamic acid (Glu/E) (g/100g protein)",
                                                                      "Glycine (Gly/G) (g/100g protein)",
                                                                      "Histidine  (His/H) (g/100g protein)",
                                                                      "Isoleucine (Ile/I) (g/100g protein)",
                                                                      "Leucine (Leu/L) (g/100g protein)",
                                                                      "Lysine (Lys/K) (g/100g protein)",
                                                                      "Methionine (Met/M) (g/100g protein)",
                                                                      "Phenylalanine (Phe/F) (g/100g protein)",
                                                                      "Proline (Pro/P) (g/100g protein)",
                                                                      "Serine (Ser/S) (g/100g protein)",
                                                                      "Threonine (Thr/T) (g/100g protein)",
                                                                      "Valine (Val/V) (g/100g protein)", "Linoleic acid (18:3, w6) (g)", 
                                                                      "Polyunsaturated fatty acids (PUFA) (g)", "Ash (g)", "Energy (kcal)",
                                                                      "Monounsaturated fatty acids (MUFA) (g)", "Phosphorus (mg)", "Sucrose (g)")]

df_theoretical_nutrients_aminoacid_environmental_data_PCA_sel1 <- df_theoretical_nutrients_aminoacid_environmental_data_PCA[c("protein_source", "Moisture_Level", selected_variables_1)]

PCA_analysis(df_theoretical_nutrients_aminoacid_environmental_data_PCA_sel1, "Moisture_Level", "protein_source", "PCA theoretical nutrients, amino acids and environmental (excluded low variance)", file_dir = NULL)

#     >>> Compute contribution of each variable to the new space of PCs and select the 10 most relevant

results_variable_contributions <- pca_variable_contributions(df_theoretical_nutrients_aminoacid_environmental_data_PCA_sel1,
                                                             method_cor = "pearson",
                                                             impute_median = TRUE,
                                                             var_expl_acum = 0.90,   # usa los PCs necesarios hasta el 90% de varianza acumulada
                                                             top_n = 12)

df_theoretical_nutrients_aminoacid_environmental_data_PCA_sel2 <- df_theoretical_nutrients_aminoacid_environmental_data_PCA_sel1[c("protein_source", "Moisture_Level", names(results_variable_contributions$top_vars), "Protein (g)", "Climate change (kg CO2 eq)")]
df_theoretical_nutrients_aminoacid_environmental_data_PCA_sel2 <- within(df_theoretical_nutrients_aminoacid_environmental_data_PCA_sel2,rm("Fructose (g)", "Tyrosine (Tyr/Y) (g/100g protein)"))


PCA_analysis(df_theoretical_nutrients_aminoacid_environmental_data_PCA_sel2, "Moisture_Level", "protein_source", "PCA theoretical nutrients, amino acids and environmental (PCA high contribution variables and Protein)", file_dir = NULL)



# =============================================
#   PCA WITH NUTRIENTS + AMINO ACIDS MODEL DATA
#     2- Principal Component Analysis. 
#         Standardise data (Center and Scale)
#
#     THE WHOLE DATASET OF PROTEIN ITEMS
# =============================================


# 2- Principal Component Analysis.
df_nutrients_aminoacids_data_model_PCA <- df_nutrients_aminoacids_data_model[,!duplicated(colnames(df_nutrients_aminoacids_data_model))]
colnames(df_nutrients_aminoacids_data_model_PCA)[57] <- "Moisture_Level"
      # With Moisture
df_nutrients_aminoacids_data_model_PCA_with_moisture = within(df_nutrients_aminoacids_data_model_PCA, rm('Moisture Level Num', 'category', 'id', 'protein_source_or_food_product', 'source_type'))
df_nutrients_aminoacids_data_model_PCA_with_moisture$Moisture_Level <- factor(df_nutrients_aminoacids_data_model_PCA_with_moisture$Moisture_Level, levels=c('Very High', 'High', 'Intermediate', 'Low', 'Very Low'))
PCA_analysis(df_nutrients_aminoacids_data_model_PCA_with_moisture, "Moisture_Level", "protein_source", "PCA nutrients and amino acids data model with moisture", file_dir = NULL)

      # With Source Type
df_nutrients_aminoacids_data_model_PCA_with_source_type = within(df_nutrients_aminoacids_data_model_PCA, rm('Moisture Level Num', 'category', 'id', 'protein_source_or_food_product', 'Moisture_Level'))
PCA_analysis(df_nutrients_aminoacids_data_model_PCA_with_source_type, "source_type", "protein_source", "PCA nutrients and amino acids data model with source type", file_dir = NULL)

#     >>> Exclude variables with a variance close to zero
numeric_variables <- unlist(lapply(df_nutrients_aminoacids_data_model_PCA, is.numeric), use.names = FALSE)
variables_var <- sapply(df_nutrients_aminoacids_data_model_PCA[,numeric_variables], var)
selected_variables <- names(variables_var[!variables_var < 0.001])
df_nutrients_aminoacids_data_model_PCA <- df_nutrients_aminoacids_data_model_PCA[c("protein_source", "Moisture_Level", "source_type",selected_variables)]

      # With Moisture
df_nutrients_aminoacids_data_model_PCA_with_moisture = within(df_nutrients_aminoacids_data_model_PCA, rm('source_type', 'Moisture Level Num'))
df_nutrients_aminoacids_data_model_PCA_with_moisture$Moisture_Level <- factor(df_nutrients_aminoacids_data_model_PCA_with_moisture$Moisture_Level, levels=c('Very High', 'High', 'Intermediate', 'Low', 'Very Low'))
PCA_analysis(df_nutrients_aminoacids_data_model_PCA_with_moisture, "Moisture_Level", "protein_source", "PCA nutrients and amino acids data model with moisture (excluded low variance)", file_dir = NULL)

      # With Source Type
df_nutrients_aminoacids_data_model_PCA_with_source_type = within(df_nutrients_aminoacids_data_model_PCA, rm('Moisture_Level', 'Moisture Level Num'))
PCA_analysis(df_nutrients_aminoacids_data_model_PCA_with_source_type, "source_type", "protein_source", "PCA nutrients and amino acids data model with source type (excluded low variance)", file_dir = NULL)



#     >>> Avoid high correlated variables
corr_matrix_nutrients_aminoacids_data_model_PCA <- cor(df_nutrients_aminoacids_data_model_PCA[,selected_variables], method="pearson")
high_corr_paired_nutrients_aminoacid_vars <- which(corr_matrix_nutrients_aminoacids_data_model_PCA > 0.85 | corr_matrix_nutrients_aminoacids_data_model_PCA < -0.85, arr.ind = TRUE) # >>>> There is no high correlated pair of variables despite the diagonal 

for (row in 1:(length(high_corr_paired_nutrients_aminoacid_vars)/2)){
  if (high_corr_paired_nutrients_aminoacid_vars[row,1] != high_corr_paired_nutrients_aminoacid_vars[row,2]){
    print(paste0(paste0(colnames(corr_matrix_nutrients_aminoacids_data_model_PCA)[high_corr_paired_nutrients_aminoacid_vars[row,1]],' - '),colnames(corr_matrix_nutrients_aminoacids_data_model_PCA)[high_corr_paired_nutrients_aminoacid_vars[row,2]]))
    print(corr_matrix_nutrients_aminoacids_data_model_PCA[high_corr_paired_nutrients_aminoacid_vars[row,1], high_corr_paired_nutrients_aminoacid_vars[row,2]])
  }
}

selected_variables_1 <- selected_variables[!selected_variables %in% c("Moisture Level Num", "Glycine (Gly/G) (g/100 g Protein)",
                                                                      "Isoleucine (Ile/I) (g/100 g Protein)", "Leucine (Leu/L) (g/100 g Protein)",
                                                                      "Lysine (Lys/K) (g/100 g Protein)", "Threonine (Thr/T) (g/100 g Protein)",
                                                                      "Serine (Ser/S) (g/100 g Protein)", "Alanine (Ala/A) (g/100 g Protein)",
                                                                      "Arginine  (Arg/R) (g/100 g Protein)", "Histidine  (His/H) (g/100 g Protein)"
                                                                      )]

        # With Moisture
df_nutrients_aminoacids_data_model_PCA_sel1_with_moisture <- df_nutrients_aminoacids_data_model_PCA[c("protein_source", "Moisture_Level", selected_variables_1)]
df_nutrients_aminoacids_data_model_PCA_sel1_with_moisture$Moisture_Level <- factor(df_nutrients_aminoacids_data_model_PCA_sel1_with_moisture$Moisture_Level, levels=c('Very High', 'High', 'Intermediate', 'Low', 'Very Low'))
PCA_analysis(df_nutrients_aminoacids_data_model_PCA_sel1_with_moisture, "Moisture_Level", "protein_source", "PCA nutrients and amino acids data model with moisture (excluded low variance)", file_dir = NULL)

        # With Source Type
df_nutrients_aminoacids_data_model_PCA_sel1_with_source_type <- df_nutrients_aminoacids_data_model_PCA[c("protein_source", "source_type", selected_variables_1)]
PCA_analysis(df_nutrients_aminoacids_data_model_PCA_sel1_with_source_type, "source_type", "protein_source", "PCA nutrients and amino acids data model with source type (excluded low variance)", file_dir = NULL)


#     >>> Compute contribution of each variable to the new space of PCs and select the 10 most relevant
        # With Moisture
results_variable_contributions <- pca_variable_contributions(df_nutrients_aminoacids_data_model_PCA_sel1_with_moisture,
                                                             method_cor = "pearson",
                                                             impute_median = TRUE,
                                                             var_expl_acum = 0.75,   # usa los PCs necesarios hasta el 90% de varianza acumulada
                                                             top_n = 12)

df_theoretical_nutrients_aminoacid_data_PCA_sel2 <- df_nutrients_aminoacids_data_model_PCA_sel1_with_moisture[c("protein_source", "Moisture_Level", names(results_variable_contributions$top_vars), "Protein (g)")]


PCA_analysis(df_theoretical_nutrients_aminoacid_data_PCA_sel2, "Moisture_Level", "protein_source", "PCA nutrients and amino acids data model with moisture (PCA high contribution variables and Protein)", file_dir = NULL)

        # With Source Type
results_variable_contributions <- pca_variable_contributions(df_nutrients_aminoacids_data_model_PCA_sel1_with_source_type,
                                                             method_cor = "pearson",
                                                             impute_median = TRUE,
                                                             var_expl_acum = 0.75,   # usa los PCs necesarios hasta el 90% de varianza acumulada
                                                             top_n = 12)

df_theoretical_nutrients_aminoacid_data_PCA_sel2 <- df_nutrients_aminoacids_data_model_PCA_sel1_with_source_type[c("protein_source", "source_type", names(results_variable_contributions$top_vars), "Protein (g)")]


PCA_analysis(df_theoretical_nutrients_aminoacid_data_PCA_sel2, "source_type", "protein_source", "PCA nutrients and amino acids data model with source type (PCA high contribution variables and Protein)", file_dir = NULL)


# ====================================================================
#   PCA WITH NUTRIENTS + AMINO ACIDS MODEL DATA + ENVIRONMENTAL DATA
#     2- Principal Component Analysis. 
#         Standardise data (Center and Scale)
#
#     THE WHOLE DATASET OF PROTEIN ITEMS
# ====================================================================


# 2- Principal Component Analysis.
df_nutrients_aminoacids_environmental_data_model_PCA <- df_nutrients_aminoacids_environmental_data_model[,!duplicated(colnames(df_nutrients_aminoacids_environmental_data_model))]
colnames(df_nutrients_aminoacids_environmental_data_model_PCA)[54] <- "Moisture_Level"
      # With Moisture
df_nutrients_aminoacids_environmental_data_model_PCA_with_moisture = within(df_nutrients_aminoacids_environmental_data_model_PCA, rm('Moisture Level Num', 'category', 'id', 'protein_source_or_food_product', 'source_type'))
df_nutrients_aminoacids_environmental_data_model_PCA_with_moisture$Moisture_Level <- factor(df_nutrients_aminoacids_environmental_data_model_PCA_with_moisture$Moisture_Level, levels=c('Very High', 'High', 'Intermediate', 'Low', 'Very Low'))
PCA_analysis(df_nutrients_aminoacids_environmental_data_model_PCA_with_moisture, "Moisture_Level", "protein_source", "PCA nutrients, amino acids and environmental data model with moisture", file_dir = NULL)

      # With Source Type
df_nutrients_aminoacids_environmental_data_model_PCA_with_source_type = within(df_nutrients_aminoacids_environmental_data_model_PCA, rm('Moisture Level Num', 'category', 'id', 'protein_source_or_food_product', 'Moisture_Level'))
PCA_analysis(df_nutrients_aminoacids_environmental_data_model_PCA_with_source_type, "source_type", "protein_source", "PCA nutrients, amino acids and environmental data model with source type", file_dir = NULL)

#     >>> Exclude variables with a variance close to zero
numeric_variables <- unlist(lapply(df_nutrients_aminoacids_environmental_data_model_PCA, is.numeric), use.names = FALSE)
variables_var <- sapply(df_nutrients_aminoacids_environmental_data_model_PCA[,numeric_variables], var)
selected_variables <- names(variables_var[!variables_var < 0.001])
df_nutrients_aminoacids_environmental_data_model_PCA <- df_nutrients_aminoacids_environmental_data_model_PCA[c("protein_source", "Moisture_Level", "source_type",selected_variables)]

      # With Moisture
df_nutrients_aminoacids_environmental_data_model_PCA_with_moisture = within(df_nutrients_aminoacids_environmental_data_model_PCA, rm('source_type', 'Moisture Level Num'))
df_nutrients_aminoacids_environmental_data_model_PCA_with_moisture$Moisture_Level <- factor(df_nutrients_aminoacids_environmental_data_model_PCA_with_moisture$Moisture_Level, levels=c('Very High', 'High', 'Intermediate', 'Low', 'Very Low'))
PCA_analysis(df_nutrients_aminoacids_environmental_data_model_PCA_with_moisture, "Moisture_Level", "protein_source", "PCA nutrients, amino acids and environmental data model with moisture (excluded low variance)", file_dir = NULL)

      # With Source Type
df_nutrients_aminoacids_environmental_data_model_PCA_with_source_type = within(df_nutrients_aminoacids_environmental_data_model_PCA, rm('Moisture_Level', 'Moisture Level Num'))
PCA_analysis(df_nutrients_aminoacids_environmental_data_model_PCA_with_source_type, "source_type", "protein_source", "PCA nutrients, amino acids and envrionmental data model with source type (excluded low variance)", file_dir = NULL)



#     >>> Avoid high correlated variables
corr_matrix_nutrients_aminoacids_environmental_data_model_PCA <- cor(df_nutrients_aminoacids_environmental_data_model_PCA[,selected_variables], method="pearson")
high_corr_paired_nutrients_aminoacid_environmental_vars <- which(corr_matrix_nutrients_aminoacids_environmental_data_model_PCA > 0.85 | corr_matrix_nutrients_aminoacids_environmental_data_model_PCA < -0.85, arr.ind = TRUE) # >>>> There is no high correlated pair of variables despite the diagonal 

for (row in 1:(length(high_corr_paired_nutrients_aminoacid_environmental_vars)/2)){
  if (high_corr_paired_nutrients_aminoacid_environmental_vars[row,1] != high_corr_paired_nutrients_aminoacid_environmental_vars[row,2]){
    print(paste0(paste0(colnames(corr_matrix_nutrients_aminoacids_environmental_data_model_PCA)[high_corr_paired_nutrients_aminoacid_environmental_vars[row,1]],' - '),colnames(corr_matrix_nutrients_aminoacids_environmental_data_model_PCA)[high_corr_paired_nutrients_aminoacid_environmental_vars[row,2]]))
    print(corr_matrix_nutrients_aminoacids_environmental_data_model_PCA[high_corr_paired_nutrients_aminoacid_environmental_vars[row,1], high_corr_paired_nutrients_aminoacid_environmental_vars[row,2]])
  }
}

selected_variables_1 <- selected_variables[!selected_variables %in% c("Moisture Level Num", "Glycine (Gly/G) (g/100 g Protein)",
                                                                      "Isoleucine (Ile/I) (g/100 g Protein)", "Leucine (Leu/L) (g/100 g Protein)",
                                                                      "Lysine (Lys/K) (g/100 g Protein)", "Threonine (Thr/T) (g/100 g Protein)",
                                                                      "Serine (Ser/S) (g/100 g Protein)", "Alanine (Ala/A) (g/100 g Protein)",
                                                                      "Arginine  (Arg/R) (g/100 g Protein)", "Histidine  (His/H) (g/100 g Protein)"
                                                                      )]

        # With Moisture
df_nutrients_aminoacids_environmental_data_model_PCA_sel1_with_moisture <- df_nutrients_aminoacids_environmental_data_model_PCA[c("protein_source", "Moisture_Level", selected_variables_1)]
df_nutrients_aminoacids_environmental_data_model_PCA_sel1_with_moisture$Moisture_Level <- factor(df_nutrients_aminoacids_environmental_data_model_PCA_sel1_with_moisture$Moisture_Level, levels=c('Very High', 'High', 'Intermediate', 'Low', 'Very Low'))
PCA_analysis(df_nutrients_aminoacids_environmental_data_model_PCA_sel1_with_moisture, "Moisture_Level", "protein_source", "PCA nutrients, amino acids and environmental data model with moisture (excluded low variance)", file_dir = NULL)

        # With Source Type
df_nutrients_aminoacids_environmental_data_model_PCA_sel1_with_source_type <- df_nutrients_aminoacids_environmental_data_model_PCA[c("protein_source", "source_type", selected_variables_1)]
PCA_analysis(df_nutrients_aminoacids_environmental_data_model_PCA_sel1_with_source_type, "source_type", "protein_source", "PCA nutrients, amino acids and environmental data model with source type (excluded low variance)", file_dir = NULL)


#     >>> Compute contribution of each variable to the new space of PCs and select the 10 most relevant
        # With Moisture
results_variable_contributions <- pca_variable_contributions(df_nutrients_aminoacids_environmental_data_model_PCA_sel1_with_moisture,
                                                             method_cor = "pearson",
                                                             impute_median = TRUE,
                                                             var_expl_acum = 0.75,   # usa los PCs necesarios hasta el 90% de varianza acumulada
                                                             top_n = 12)

df_nutrients_aminoacids_environmental_data_model_PCA_sel2 <- df_nutrients_aminoacids_environmental_data_model_PCA_sel1_with_moisture[c("protein_source", "Moisture_Level", names(results_variable_contributions$top_vars), "Protein (g)", "Climate change (kg CO2 eq)")]


PCA_analysis(df_nutrients_aminoacids_environmental_data_model_PCA_sel2, "Moisture_Level", "protein_source", "PCA nutrients, amino acids and envrionmental data model with moisture (PCA high contribution variables and Protein)", file_dir = NULL)

        # With Source Type
results_variable_contributions <- pca_variable_contributions(df_nutrients_aminoacids_environmental_data_model_PCA_sel1_with_source_type,
                                                             method_cor = "pearson",
                                                             impute_median = TRUE,
                                                             var_expl_acum = 0.75,   # usa los PCs necesarios hasta el 90% de varianza acumulada
                                                             top_n = 12)

df_nutrients_aminoacids_environmental_data_model_PCA_sel2 <- df_nutrients_aminoacids_environmental_data_model_PCA_sel1_with_source_type[c("protein_source", "source_type", names(results_variable_contributions$top_vars), "Protein (g)", "Climate change (kg CO2 eq)")]


PCA_analysis(df_nutrients_aminoacids_environmental_data_model_PCA_sel2, "source_type", "protein_source", "PCA nutrients, amino acids and envrionmental data model with source type (PCA high contribution variables and Protein)", file_dir = NULL)




