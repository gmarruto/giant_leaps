data_availability <- function(df_var1, df_var2, plot_title, xlabel = NULL, ylabel = NULL, file_dir = NULL){
  
  contingency_tbl <- table(df_var1, df_var2)
  df_contingency_tbl <- as.data.frame(contingency_tbl)
  df_contingency_tbl$hasValue <- ifelse(df_contingency_tbl$Freq > 0, 1, 0)
  
  availability_plot <- ggplot(df_contingency_tbl, aes(x = df_var2, y = df_var1, fill = hasValue)) +
    geom_tile() +
    scale_fill_gradient(low = "#dde8bf", high = "#76a200") +
    theme_minimal() +
    labs(x = xlabel, y = ylabel) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),  # Ajustar el tamaño y la rotación de las etiquetas del eje x
          axis.text.y = element_text(size = 8))  # Ajustar el tamaño de las etiquetas del eje y

  ggsave(paste0(file_dir, plot_title,".png"), plot = availability_plot, width = 10, height = 8)
  
  return(contingency_tbl)

}


PCA_analysis <- function(df, var_colour, var_shape, plot_title, file_dir = NULL){

  is_numeric <- sapply(df, is.numeric)
  
  # Filter dataframe by numeric columns and columns with no NA
  # and variance different to 0
  
  df_PCA <- df[,is_numeric][colSums(is.na(df[,is_numeric])) == 0]
  df_PCA <- df_PCA[ , which(apply(df_PCA, 2, var) != 0)]
  
  pc <- prcomp(df_PCA, center = TRUE, scale = TRUE)
  
  if(is.na(var_shape)){
    biplot_PCA_plot <- autoplot(pc, which = "biplot", data = df, 
                                colour = var_colour, 
                                size = 3, loadings = TRUE, loadings.label = TRUE, loadings.label.size = 3)
  } else{
    biplot_PCA_plot <- autoplot(pc, which = "biplot", data = df, 
                                colour = var_colour, shape = var_shape, 
                                size = 3, loadings = TRUE, loadings.label = TRUE, loadings.label.size = 3) +
      scale_shape_manual(values=seq(0,length(unlist(unique(df[var_shape])))))
    #scale_color_manual(values = c("#be6e5a", "#324ba0", "#5abe6e"))
  }
  
  
  ggsave(paste0(file_dir,plot_title,"_biplot.png"), plot = biplot_PCA_plot, width = 10, height = 8)
  
  
  loadings1 <- pc$rotation[,1]
  df_loadings1 <- data.frame(
    Variable = names(loadings1),
    Loading = loadings1
  )
  
  loading_PCA1_plot <- ggplot(df_loadings1, aes(x = Loading, y = reorder(Variable, Loading))) +
    geom_bar(stat = "identity", fill = "skyblue") +
    labs(title = paste0("Principal component ",toString(1)),
         x = "Loading",
         y = "Variable") +
    theme_minimal() + 
    theme(
      plot.title = element_text(hjust = 0.5)
    )
  
  loadings2 <- pc$rotation[,2]
  df_loadings2 <- data.frame(
    Variable = names(loadings2),
    Loading = loadings2
  )
  
  loading_PCA2_plot <- ggplot(df_loadings2, aes(x = Loading, y = reorder(Variable, Loading))) +
    geom_bar(stat = "identity", fill = "skyblue") +
    labs(title = paste0("Principal component ",toString(2)),
         x = "Loading",
         y = "Variable") +
    theme_minimal() + 
    theme(
      plot.title = element_text(hjust = 0.5)
    )
  
  
  loadings_PCA_plot <- grid.arrange(loading_PCA1_plot, loading_PCA2_plot, ncol = 2)

  
  ggsave(paste0(plot_title,"_loadings.png"), plot = loadings_PCA_plot, width = 10, height = 8)
  
  return(list(loadings=pc$rotation, center = pc$center, scale=pc$scale))
}


pca_variable_contributions <- function(df,
                                       vars = NULL,                # si NULL, usa numéricas
                                       method_cor = c("pearson","spearman"),
                                       impute_median = TRUE,       # imputación simple para NA
                                       var_expl_acum = 0.90,       # varianza acumulada objetivo
                                       n_components = NULL,        # si quieres fijar K
                                       top_n = 15,                 # nº de variables a devolver
                                       quiet = FALSE) {
  method_cor <- match.arg(method_cor)
  
  # 0) Selección de columnas
  X <- df
  if (!is.null(vars)) {
    faltan <- setdiff(vars, names(df))
    if (length(faltan)) stop("Variables no encontradas: ", paste(faltan, collapse=", "))
    X <- df[vars]
  }
  
  # 1) Mantener sólo numéricas
  num_idx <- sapply(X, is.numeric)
  X <- X[ , num_idx, drop = FALSE]
  if (ncol(X) < 2) stop("Se requieren al menos 2 variables numéricas.")
  
  # 2) Imputación simple (mediana) si se desea
  if (impute_median) {
    X <- as.data.frame(lapply(X, function(col) {
      ifelse(is.na(col), median(col, na.rm = TRUE), col)
    }))
  } else {
    # prcomp omite filas con NA si na.action=na.omit; aquí imputamos para conservar todo
    if (anyNA(X) && !quiet) message("Hay NA y no se imputan: prcomp podría omitir filas.")
  }
  
  # 3) PCA (centrado + escalado)
  pca <- prcomp(X, center = TRUE, scale. = TRUE, retx = TRUE)
  ev  <- pca$sdev^2
  evr <- ev / sum(ev)
  evr_cum <- cumsum(evr)
  
  # 4) Elegir K componentes
  if (!is.null(n_components)) {
    K <- max(1, min(n_components, ncol(X)))
  } else {
    K <- which(evr_cum >= var_expl_acum)[1]
  }
  if (!quiet) {
    message(sprintf("Componentes retenidos (K) = %d | Varianza acumulada = %.1f%%",
                    K, 100*evr_cum[K]))
  }
  
  # 5) Cargas (loadings) y contribución ponderada
  # pca$rotation: columnas = PCs, filas = variables
  L <- pca$rotation[ , 1:K, drop = FALSE]        # loadings (variables x K)
  contrib <- as.vector((L^2) %*% matrix(evr[1:K], ncol = 1)) # (variables x 1)
  names(contrib) <- colnames(df)[3:length(colnames(df))]
  
  ranking <- sort(contrib, decreasing = TRUE)
  top_n <- min(top_n, length(ranking))
  top_vars <- head(ranking, top_n)
  
  list(
    ranking = ranking,              # todas las variables ordenadas
    top_vars = top_vars,            # top-N
    K = K,                          # nº de PCs usados
    evr = evr,                      # varianza explicada por PC
    evr_cum = evr_cum,              # varianza acumulada
    pca = pca                       # objeto prcomp para gráficas adicionales
  )
}




connect_PostgreSQL <- function(db_conn_file, db_name = "giantleaps"){
  db_conn_info <- fromJSON(file=db_conn_file)
  host <- db_conn_info[[db_name]]['host']
  port <- db_conn_info[[db_name]]['port']
  db <- db_conn_info[[db_name]]['db']
  pw <- db_conn_info[[db_name]]['password']
  user <- db_conn_info[[db_name]]['user']
  
  postgres_driver <- dbDriver('PostgreSQL')
  return(dbConnect(postgres_driver, dbname = db, host = host, port = port, user = user, password = pw))
}
