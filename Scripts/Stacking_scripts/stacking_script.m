%% Now we combine models using Ridge Regressione
clear;clc;
% Load and store the variables we aim to predict
load vars_clean.mat
vars = vars_clean(:,11:44); % define which variables we want to predict
load twins_clean.mat
twins = twins;

%%% I SHOULD DO THE VARYING STATES BECAUSE THAT IS WHEN IT WILL BE MORE
%%% INTERESTING
% Load and store our current predictions
% load FC_HMM_zeromean_1_covtype_full_stacked/HMM_predictions_stack.mat
% vars_hat = vars_hat_ST_reps;
load FC_HMM_zeromean_1_covtype_full_stack_ALL_vs/HMM_predictions_stack.mat
vars_hat = vars_hat_ST_reps; % note the current predictions (as produced by predictPhenotype)
% states = [3 8 13 18 23]

%% Explore data
figure()
bar(mean(explained_variance_ST_reps,1)')
xlabel('No. states'); ylabel('Mean r^2')
title('Mean r^2 across intelligence variables by no. states of HMM runs')
K_vec = {'3','8','13','18','23'};
set(gca, 'XTick', 1:length(K_vec),'XTickLabel',K_vec);


figure()
bar([3 8 13 18 23], mean(squeeze(mean(W,2)),2))
xlabel('No. states'); ylabel('Contribution of prediction');
title('Contribution of original HMM repetition to the stacked prediction')




%%

% Set up metafunctions - just using the first (constant) metafeature is
% equivalent to using no metafeatures
load HMMs_meta_data.mat
n_subjects = size(vars,1); % number of subjects
MF1 = ones(n_subjects,5);
MF2 = maxFO_all_reps./10;
MF3 = switchingRate_all_reps;
metafeatures = [MF1 MF2 MF3];
options = struct();

% Perform stacked regression & stacked regression with metafeatures
[vars_hat_st,vars_hat_FWLS,explained_variance_st,explained_variance_FWLS,weights,weights_FWLS,folds] = stack_regress_metaf_V2(vars,vars_hat,options,metafeatures,twins);


%%
% Plot the original 5 repetitions and the new combined prediction
X = 1:size(vars,2);
figure
scatter(X,explained_variance_st,'o','g'); hold on; % plot my stacked value
%scatter(X,explained_variance_FWLS,'o','r') % plot my stacked value with metafeature
scatter(X,explained_variance_ST,'o','k') % plot Diego's stacked value
%scatter(X,explained_variance_ST_reps(:,reps),'x','b') % plot the original 5 repetitions of HMM
scatter(X,explained_variance_ST_reps,'x','b') % plot the original 5 repetitions of HMM

% Format chart
xlabel('Intelligence features'); ylabel('r^2')
legend('Stacked predictor','FWLS predictor','Diego stacked predictor','Original HMM runs')


% Old metafeatures
% MF1 = ones(n_subjects,1);
% MF2 = mean(maxFO_all_reps,2)/10;
% MF3 = mean(switchingRate_all_reps,2)*3;


%% Let's reproduce Diego's stacking results





