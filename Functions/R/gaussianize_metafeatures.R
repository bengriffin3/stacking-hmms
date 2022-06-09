#! /usr/local/bin/env RScript

#  The first line tells terminal to:
# execute with a RScript interpreter, using the env program search path to find it


#In this script we Gaussianize metafeatures
#install libraries
library(LambertW) # install library for Gaussinization
library(R.matlab) # library to read matlab data formats into R

# read in our metadata
metadata <- readMat("/Users/bengriffin/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps/HMMs_meta_data_subject_r_1_27_40_63.mat")
#str(metadata) # check out data structure
Entropy <- metadata$Entropy.subject.all # store entropy
Likelihood <- metadata$likelihood.subject.all # store likelihood
Squared_error <- metadata$squared.error.all # store error values

# function for Gaussinaizing metafeatures
metafeature_gaussianize <- function(meta_rep) {
  meta_gauss <- Gaussianize(meta_rep, type = c("s"), return.tau.mat = TRUE)
  metafeature_gaussianized<- get_input(meta_rep, c(meta_gauss$tau.mat[, 1]))
  return(metafeature_gaussianized)
}

# initialise array for Gaussian data
Entropy_gaussianized = matrix(NaN, 1001, 45)
Likelihood_gaussianized = matrix(NaN, 1001, 45)

# Gaussianize the metafeatures
for (rep in c(1:27,41:45)){
  print(rep)
Entropy_gaussianized[,rep] <- metafeature_gaussianize(Entropy[,rep])
Likelihood_gaussianized[,rep] <- metafeature_gaussianize(Likelihood[,rep])
}



# note that we don't need to remove subjects because we have the brain imaging data
# (and hence HMM repetition data) for all out subjects, it's the variables/features
# that we don't have data for.

# either type full path name or change to directory desired
#getwd()
#filename <- paste("/Users/bengriffin/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps/HMMs_meta_data_subject_r_1_27_40_63_gaussianized.mat", sep = "")
#writeMat(filename, Entropy_gaussianized = Entropy_gaussianized, Likelihood_gaussianized = Likelihood_gaussianized)

Squared_error_gaussianized <- array(rep(NaN, 1001*5*34), c(1001, 5, 34))

# Squared errors
for (var in c(1:34)){
  print(var)
  for (rep in c(1:5)){
    # Prediction errors ### WHAT TYPE? S, SS, H?
    Squared_error_rv = Squared_error[,rep,var]
    non_nan_idx <- which(!is.na(Squared_error_rv))
    #print(length(Squared_error_rv))  # check we have 1001 elements (1 for each subjects)
    Squared_error_test <- sort(Squared_error_rv)
    Squared_error_gauss <- Gaussianize(Squared_error_test, type = c("s"), return.tau.mat = TRUE)
    Squared_error_gauss_data <- get_input(Squared_error_test, c(Squared_error_gauss$tau.mat[, 1]))  # same as out$input
    Squared_error_gauss_data_store = matrix(NaN, 1001)
    Squared_error_gauss_data_store[non_nan_idx] = Squared_error_gauss_data
    Squared_error_gaussianized[,rep,var] = Squared_error_gauss_data_store

  }
}

filename <- paste("/Users/bengriffin/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps/NEW_HMMs_meta_data_subject_r_1_27_40_63_gaussianized.mat", sep = "")
writeMat(filename, Entropy_gaussianized = Entropy_gaussianized, Likelihood_gaussianized = Likelihood_gaussianized, Squared_error_gaussianized = Squared_error_gaussianized)


