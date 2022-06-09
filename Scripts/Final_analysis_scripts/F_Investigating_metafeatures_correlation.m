%% Simulated metafeautures (investigating effect of rho)
% runtime; 66 minutes for full rho vec (to recreate images shared with Diego)
% In this script we simulate metafeatures that are correlated, with a
% specified rho, to the mean squared error of the predictions produced
% using KRR on the distance matrices, formed by using the KL divergence on
% the HMM repetitions.
% The figures produced give us an idea of the effect of hte metafeatures by
% showing us the final prediction accuracies without metafeatures, with the
% proposed genuine metafeatures, and finally the simulated metafeatures
% (from 1 to 5 additional metafeatures)

% ISSUES WITH SCRIPT
% 1) Currently, when we don't use metafeatures, I run it
% twice (once within the same function as the simulated metafeatures, and
% once within the same function as the genuine metafeatures. They have
% similar results but this obviously isn't the way to do it).
% 2) Also, I repeat the whole thing for 6 metafeatures, but that means I am
% running the 'no metafeatures' code 6 times, which is redundant.
% 3) Need to update the correlations of the real metafeature and the accuracy 
% of the HMM predictions to mean of all variables and all repetitions (see code below)

clear; clc; %close all;
rng('default') % set seed for reproducibility
%DirOut = '/Users/au699373/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_vary_states_3_14/'; % Test folder
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\Varying_states\'; % Test folder

% load data
load([ DirOut  'HMM_predictions.mat']) % load predictions
load('vars_target.mat') % load targets
load([ DirOut 'HMMs_meta_data_subject']);
load([ DirOut 'HMMs_meta_data_GROUP']); % load metadata
%load([ DirOut 'HMMs_meta_data_subject_gaussianized']) % load Gaussianized metadata


%[Entropy_gaussianized,Likelihood_gaussianized] = gaussianize_r(); % Gaussianize metafeatures (using script in R) (note: must change the save directory to the correct folder in R)

% Note dimensions of arrays
n_var = size(vars_hat,2);
n_repetitions = size(vars_hat,3);

% Select simulation choices
simulation_options.rep_selection = 1:n_repetitions;% Select repetitions to use for simulation (default is all repetitions) % [2 6 14 25 44];
simulation_options.n_folds = 10;
simulation_options.corr_specify = []; % specify correlation between simulated metafeature and accuracy of HMM prediction (if empty, we use same correlation as true metafeatures)
simulation_options.simulated_metafeatures_in_r = [];

% Select metafeature e.g. Entropy/Likelihood
metafeature_store = [metastate_profile likelihood_subject Entropy_subject_log2 K_all_reps maxFO_all_reps switchingRate_all_reps];
n_metafeatures = size(metafeature_store,2)/n_repetitions;

% Select correlation of simulated metafeatures and accuracy of HMM repetition predictions
rho_vec = 0:0.005:0.995;
%rho_vec = [0.25 0.3];% run a quicker version where we only look at a limited no. of rho values

% initialise variables
ev_rho_store = NaN(length(rho_vec),n_metafeatures);
mpe_rho_store = NaN(length(rho_vec),n_metafeatures);
no_metafeature_mse = NaN(n_metafeatures,1);
no_metafeature_ev = NaN(n_metafeatures,1);

% Note, with more than 1 metafeature we get a warning about a singular
% matrix. Ignoring for now but should look into.
for j = 1:size(metafeature_store,2)/n_repetitions
    sprintf('Additional metafeatures: %i',j)
    for i = 1:length(rho_vec)
        rho = rho_vec(i) % for each simulated rho
        simulation_options.corr_specify = rho;
        metafeature = metafeature_store(:,1:n_repetitions*j); % take an increasing number of metafeatures
        
        [pred_ST,pred_stack_all,MSE_stack_ST,MSE_stack_all,EV_ST,EV_all,pearson_error_metafeature] = simulation_full(vars,vars_hat,metafeature,simulation_options);
        % _ST = 5 repetitions

        % Accuracy of predictions across all variables using simulated metafeatures
        mpe_rho_store(i,j) = sum(MSE_stack_all(:,4,1));
        ev_rho_store(i,j) = sum(EV_all(:,4,1));

    end
    % Accuracy of predictions across all variables using no metafeatures (this doesn't need to be done for each new metafeature - see issues at top of doc)
    no_metafeature_mse(j) = sum(MSE_stack_all(:,2,2)); % could take MSE_stack_all(:,2,1) instead (see issue at top of doc)
    no_metafeature_ev(j) = sum(EV_all(:,2,2)); % could take EV_all(:,2,1) instead (see issue at top of doc)
end

save('accuracy_by_rho.mat','mpe_rho_store','ev_rho_store','no_metafeature_mse','no_metafeature_ev')


%% Note down accuracies & correlations when using true metafeature 
% Again, this is the true metafeature, which is calculated everytime we
% add a new metafeatures, but doesn't need to be, so this needs to be
% sorted out (see issues at top of scripts). For now, we just take the
% latest version of 'MSE_stack_all' and 'EV_all', but we could use any of
% the 5 made as we added more metafeatures
true_metafeature_mse = sum(MSE_stack_all(:,4,2));
true_metafeature_ev = sum(EV_all(:,4,2));


% To get the correlations of the real metafeature and the accuracy of the HMM predictions,
% we just take an example correlation from one of the variables 
% NEED TO UPDATE THIS TO MAYBE MEAN OF ALL VARIABLES AND OF ALL REPETITIONS IF SHARING?
% Remember, the simulated metafeature has a specified correlation that is the same for all variables and all repetitions (since that is the way it is simualated), 
% but for the real metafeature we have a correlation vlaue that is different for each variable and each repetition - hence update to  mean?
squared_error = (vars - vars_hat).^2;
true_corr = corr(likelihood_subject,squeeze(squared_error(:,1,:))); 
true_corr = abs(true_corr(1,1));
% I want stacked ridge without metafaetue, then real metafeature, then simulated
%% PLOT SIMULATIONS
% 2-dim: (1) LSQ (no mf) (2) Ridge (no mf) (3) FWLS LSQ (4) FWLS Ridge
% 3-dim: (1) Simulated (2) True metafeature

% Note that currently 

X = rho_vec;
figure();
Y = ev_rho_store;
%Y_smooth = reshape(smooth(ev_rho_store),[200 6]);
%Y_smooth_clean = [ev_rho_store(1:2,:); Y_smooth(3:end-2,:);  ev_rho_store(end-1:end,:)];
%Y = Y_smooth_clean;
plot(X,Y); hold on % simulated metafeature
scatter(true_corr,true_metafeature_ev,'x','r') % real metafeature
scatter(0,no_metafeature_ev(1),'r') % no metafeature (we take no_metafeature_ev(1) but could take any element - see issues at top of doc regarding 'no metafeatures' redundancy)
title('Accuracy of stacked predictions when simulating {\itn} metafeatures');
xlabel('Correlation of simulated metafeature and accuracy of HMM run predictions, \rho');
ylabel('Total explained variance (across 34 variables)');
legend('FWLS Ridge (simulated metafeature n = 1)','FWLS Ridge (simulated metafeature n = 2)','FWLS Ridge (simulated metafeature n = 3)','FWLS Ridge (simulated metafeature n = 4)','FWLS Ridge (simulated metafeature n = 5)','FWLS Ridge (simulated metafeature n = 6)','FWLS Ridge (true metafeature)','Stacked Ridge (without metafeatures)');

%%
Y = mpe_rho_store;
figure(); plot(X,Y); hold on % simulated metafeature
scatter(true_corr,true_metafeature_mse,'x','r') % real metafeature
scatter(0,no_metafeature_mse(1),'r')  % no metafeature (we take no_metafeature_mse(1) but could take any element - see issues at top of doc regarding 'no metafeatures' redundancy)
title('Accuracy of stacked predictions when simulating {\itn} metafeatures');
xlabel('Correlation of simulated metafeature and accuracy of HMM run predictions, \rho');
ylabel('Total mean squared error (across 34 variables)');
legend('FWLS Ridge (simulated metafeature n = 1)','FWLS Ridge (simulated metafeature n = 2)','FWLS Ridge (simulated metafeature n = 3)','FWLS Ridge (simulated metafeature n = 4)','FWLS Ridge (simulated metafeature n = 5)','FWLS Ridge (simulated metafeature n = 6)','FWLS Ridge (true metafeature)','Stacked Ridge (without metafeatures)');

%%
figure()
scatter(1:34,EV_all(:,4,1))




