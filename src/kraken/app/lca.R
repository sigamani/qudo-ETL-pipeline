library(poLCA)
library(jsonlite)
library(doParallel)
library(depmixS4)
library(foreach)
library(dplyr)


# Get the number of cores to use for parallel processing
registerDoParallel(cores=detectCores())

# Run the LCA analysis in parallel mode
parallel_latent_class <- function(data, k, formula) {
      foreach(i=3:k, .packages="poLCA", .errorhandling = 'remove') %dopar%
        poLCA(formula, data, nclass=i, nrep = 100, verbose = FALSE)
  }


lca <- function(data, vars_to_include_json){
  set.seed(99)
# Loads of data processing to get the data JSON into the right shape
  # Remove any non-numeric columns from data
  # data <- as.data.frame(data)
  data <- dplyr::select(data, where(is.numeric))
  # LCA only takes in positive integers, so adding 1 to the whole dataframe
  data <- data + 1
  # Set as vector to use in LCA
  variables_vector <- as.vector(sapply(vars_to_include_json, as.name))
  # Create the formula using the columns
  formula <- as.formula(paste("cbind(", paste0(variables_vector, collapse = ","),")~ 1"))
  if (ncol(data) > 100) {
    # corr_matrix <- cor(data)
    # highly_correlated <- findCorrelation(corr_matrix, cutoff = 0.10)
    # highly_correlated <- sort(highly_correlated)
    # reduced_df <- data[, -c(highly_correlated)]
    reduced_df <- data[, sample(ncol(data), max(c(70, floor(ncol(data) * 0.3))))]
    new_col_names <- names(reduced_df)
    variables_vector <- as.vector(sapply(new_col_names, as.name))
    formula <- as.formula(paste("cbind(", paste0(variables_vector, collapse = ","), ")~ 1"))
  }
  lcas <- parallel_latent_class(data, 8, formula)
  # Select the model with the lowest BIC
  min_bic <- 10000000
  for (model in lcas) {
    if (model$bic < min_bic) {
      model_to_return <- model
      min_bic <- model$bic
    }
  }
  # metrics <- get_metrics_df(data, model_to_return$predclass)
  return(list("class_labels" = model_to_return$predclass))
}

fit_mm_lca <- function(data, i, formula, data_types_list) {
  mod <- mix(formula, data = data, family = data_types_list, nstates = i)
  fmod <- fit(mod, verbose = FALSE)
  fmod
}

parallel_mixture_model <- function(
      data,
      k,
      formula,
      data_types_list) {
        foreach(i=2:k, .packages = "depmixS4", .errorhandling = 'remove') %dopar%
          fit_mm_lca(data, i, formula, data_types_list)
}

parallel_mm_lca <- function(data, k, f, data_types_list){
  lcas <- parallel_mixture_model(data, k, f, data_types_list)
}


mixture_modelling_lca <- function(data, vars_to_include_json, data_types_for_vars) {
  # Remove any non-numeric columns from data
  data <- dplyr::select(data, where(is.numeric))
  # LCA only takes in positive integers, so adding 1 to the whole dataframe
  data <- data + 1
  # Set as vector to use in LCA
  variables_vector <- as.vector(sapply(vars_to_include_json, as.name))
  # Get formula into correct list shape
  form_list <- list()
  for (col in variables_vector) {
    formula <- as.formula(paste0(rlang::as_string(col), "~1"))
    form_list <- append(form_list, formula)
  }
  model_types <- list(categorical = multinomial(), continuous = gaussian())
  data_types <- model_types[match(data_types_for_vars, names(model_types))]
  # Get models
  lcas <- parallel_mixture_model(data, 8, form_list, data_types)
  min_bic <- 1000000
  for (model in lcas) {
    model_bic <- BIC(model)
    if (model_bic < min_bic) {
      model_to_return <- model
      min_bic <- model_bic
    }
  }
  # metrics <- get_metrics_df(data, model_to_return@posterior$state)
  return(list("class_labels" = model_to_return@posterior$state))
}