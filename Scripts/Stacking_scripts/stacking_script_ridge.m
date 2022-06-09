%% Now we combine models using Ridge Regressione
clear;clc;
% Load and store the variables we aim to predict
load vars_clean.mat
vars = vars_clean(:,11:44); % define which variables we want to predict

% Load and store our current predictions
load FC_HMM_zeromean_1_covtype_full_stacked/HMM_predictions_stack.mat
vars_hat = vars_hat_ST_reps;
%load FC_HMM_zeromean_1_covtype_full_vary_states_3_repetitions/HMM_predictions_stack.mat
%vars_hat = vars_hat_ST_reps(:,:,[1 4 7 10 13]); % note the current predictions (as produced by predictPhenotype)


% Set up metafunctions - just using the first (constant) metafeature is
% equivalent to using no metafeatures
load HMMs_meta_data.mat
n_subjects = size(vars,1); % number of subjects
MF1 = ones(n_subjects,1);
MF2 = mean(maxFO_all_reps,2)*3;%/10;
MF3 = mean(switchingRate_all_reps,2)*3;
metafeatures = [MF1 MF2 MF3];

% Perform stacked regression & stacked regression with metafeatures
[vars_hat_st,vars_hat_FWLS,explained_variance_st,explained_variance_FWLS] = stack_regress_metaf(vars,vars_hat,metafeatures);

% Plot the original 5 repetitions and the new combined prediction
X = 1:size(vars,2);
figure
scatter(X,explained_variance_st,'o','g'); hold on;
scatter(X,explained_variance_FWLS,'o','r')
scatter(X,explained_variance_ST_reps,'x','b')
%scatter(X,explained_variance,'x','b')

% Format chart
xlabel('Intelligence feature'); ylabel('r^2')
legend('Stacked predictor','FWLS predictor','Original HMM runs')
