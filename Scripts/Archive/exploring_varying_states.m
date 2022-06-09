%% Assessing correlation between predictions (consistent no. of states)
% Load hmm predictions
clear;clc;
DirResults = ['Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/'];
DirFC = ['FC_HMM_zeromean_1_covtype_full_8_states/'];
load([DirResults DirFC 'HMM_predictions.mat'])

% Clean hmm predictions (remove subject where we have NaNs in predictions)
idx_nans = any(isnan(vars_hat(:,:,1)), 2);
varsKEEP = true(size(vars_hat,1),1);
varsKEEP(idx_nans) = 0;
vars_hat_hmm = vars_hat(varsKEEP,:,:); 
explained_variance_hmm = explained_variance;

% Load and clean hmm predictions (varying states)
load FC_HMM_zeromean_1_covtype_full_vary_states_3_14/HMM_predictions.mat
vars_hat_hmm_varying_states = vars_hat(varsKEEP,:,:);
explained_variance_varying_states = explained_variance;

% Load and clean hmm stacked predictions
load FC_HMM_zeromean_1_covtype_full_stacked/HMM_predictions_stack.mat
vars_hat_stack = vars_hat_ST(varsKEEP,:,:);
vars_hat_stack_reps = vars_hat_ST_reps(varsKEEP,:,:);
explained_variance_stack = explained_variance_ST;
explained_variance_stack_reps = explained_variance_ST_reps;

%Load and clean hmm stacked predictions of varying states
% load FC_HMM_zeromean_1_covtype_full_vary_states_63_reps/HMM_predictions.mat
% vars_hat_stack_states = vars_hat_ST(varsKEEP,:,:);
% vars_hat_stack_states_reps = vars_hat_ST_reps(varsKEEP,:,:);
% explained_variance_stack_states = explained_variance_ST;
% explained_variance_stack_states_reps = explained_variance_ST_reps;

% Load features then store the intelligence ones
load('feature_groupings.mat')
feature_vec = feature_groupings(2)+1:feature_groupings(3); %[1:151];

% Extract intelligence features 
load 'Behavioural Variables/vars_clean.mat'
%vars_clean = vars_clean(~idx_nans,:);
vars_intelligence = vars_clean(:,feature_vec);
%vars_hat_hmm_intelligence = vars_hat_hmm(:,feature_vec,:);








%% How good are our predictors?
% We can measure the efficacy of our predictors by looking at e.g. explained variance

% We can also measure efficacy by looking at residual sum of squares (after standardising)
vars_clean_intelligence_standardised = (vars_clean_intelligence - mean(vars_clean_intelligence))./std(vars_clean_intelligence);
LSE_1 = sum(squeeze(sum((vars_clean_intelligence - vars_hat_hmm_intelligence).^2))) % calculate MSE
% Now we could e.g. choose the best predicor
[M,I] = min(LSE_1);
I % Result: predictor 4 is the best predictor

%% Exploring repetitions with a different number of states
% We now want to vary the number of states, so that our repetitions are
% more dissimilar to each other. However, because the HMM can sometimes
% product 'bad' results, we run the HMM for 3 times for each number of
% states, and keep the best
load FC_HMM_zeromean_1_covtype_full_vary_states_3_repetitions/HMM_predictions.mat
explained_variance_vary_states_3_reps = explained_variance;
K_vec = [3 3 3 5 5 5 8 8 8 13 13 13 15 15 15 18 18 18 23 23 23 25 25 25 28 28 28 33 33 33];
reshaped_exp_var = reshape(explained_variance_vary_states_3_reps,[34,3,10]);
vars_hat = vars_hat(varsKEEP,:,:);
vars_hat_vary_states_3_reps = reshape(vars_hat,[941,34,3,10]);
% 

%% Choosing best repetition for each number of states
% For one particular state number, we plot the 3 repetitions for each
% intelligence feature and then choose the best repetition
x = 1:34;
figure
for rep = 1:3
    for k = 10
        y = reshaped_exp_var(:,rep,k);
        scatter(x,y)
        hold on
    end
end
% Format chart
title('r^2 for intelligence features, 3 HMM runs for 13 states');
xlabel('Intelligence features'); ylabel('r^2');
legend('Repetition 1','Repetition 2', 'Repetition 3')

% Repetition to keep:
% 3 states = rep 1; 5 states = rep 3; 8 states = rep 3; 13 states = rep 2
% 15 states = rep 1; 18 states = rep 3; 23 states = rep 3; 25 states = rep 2
% 28 states = rep 3; 33 states = rep 3;
best_rep = [1 3 3 2 1 3 3 2 3 3];
exp_var_best = zeros(34,10);
vars_hat_best = zeros(n_subjects,34,10);
for k = 1:10
    exp_var_best_iter = reshaped_exp_var(:,best_rep(k),k);
    exp_var_best(:,k) = exp_var_best_iter;
    vars_best_iter = vars_hat_vary_states_3_reps(:,:,best_rep(k),k);
    vars_hat_best(:,:,k) = vars_best_iter;
end
%    

