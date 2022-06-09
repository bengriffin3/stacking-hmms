%% SIMULATED DATA
clc; clear;
rng('default') % set for reproducibility
DirOut = '/Users/bengriffin/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps/'; % Test folder
%DirOut = '/Users/bengriffin/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_stack_ALL_vs_V4/'; % Test folder

% load data
load([ DirOut  'HMM_predictions.mat']) % load predictions
load([ DirOut  'vars_target.mat']) % load targets
load([ DirOut 'HMMs_meta_data_subject_r_1_27_40_63']) % load metadata    %load([ DirOut  'HMMs_meta_data_subject.mat']) % load metadata
load([ DirOut 'HMMs_meta_data_subject_r_1_27_40_63_gaussianized']) % load Gaussianized metadata

% select metafeature e.g. Entropy, Entropy (gaussianized), Likelihood, Likelihood (gaussianized)
Metafeature_all = likelihood_subject_all; %Metafeature_all = Entropy_gaussianized;

% Select the 5 repetitions we want
vars_hat = vars_hat(:,:,[2 6 14 25 44]);
Metafeature_all = Metafeature_all(:,[2 6 14 25 44]);

% normalize variables and target (should we remove outliers before normalizing since this will change max/min?
vars_norm = (vars-min(vars))./(max(vars)-min(vars));
vars_hat_norm = (vars_hat-min(vars_hat))./(max(vars_hat)-min(vars_hat));
Metafeature_all = (Metafeature_all-min(Metafeature_all))./(max(Metafeature_all)-min(Metafeature_all)); %(Update: this does nothing)
% WHY DOES NORMALIZE FUNCTION NOT WORK?
% normalize() is actually standardization (mean 0 std 1)
%vars = normalize(vars);
%vars_hat = normalize(vars_hat);

% Intialise variables
n_subjects = size(vars_hat,1); n_var = size(vars_hat,2); n_repetitions = 5; 
EV_ST = NaN(n_var,n_repetitions); EV_all = NaN(n_var,4);
MSE_stack_ST = NaN(n_var,n_repetitions); MSE_stack_all = NaN(n_var,4);
pearson_error_metafeature = NaN(n_var,n_repetitions);
pearson_error_metafeature_simulated = NaN(n_var,n_repetitions);
pred_ST = NaN(n_subjects,n_repetitions,n_var); pred_stack_all = NaN(1001,4,n_var);
Metafeature_simu = NaN(n_subjects,n_repetitions,n_var);


for var = 1:34
    
    % remove NaNs
    [vars_target,Predictions,Metafeature,squared_error_gaussianized,non_nan_idx] = nan_subject_remove(var,vars_norm,vars_hat_norm,Metafeature_all,Squared_error_gaussianized);

    % remove outliers
    %[vars_target,Predictions,Metafeature,squared_error_gaussianized] = outlier_remove(vars_target,Predictions,Metafeature,squared_error_gaussianized);

    % determine accuracy of predictions (we want our metafeature to be correlated with this)
    squared_error = (vars_target - Predictions).^2;
    
    %%%%%%%%%% METAFEATURE ARRAY SIMULATION %%%%%%%%%%%%%% 
    % we simulate our metafeature to have the same correlation as our  actual metafeature
    for rep = 1:5; [pearson_error_metafeature(var,rep), ~] = corr(squared_error(:,rep),Metafeature(:,rep),'type','pearson'); end
    p_metafeature = pearson_error_metafeature(var,:);% + 0.6; % make correlation more favourable
    
    % simulate metafeature
    Metafeature_simu(non_nan_idx,:,var) = metafeature_simulation_creation(squared_error,p_metafeature,Metafeature);
    metafeature_array = [repmat(ones(size(Metafeature)),1,1) Metafeature_simu(non_nan_idx,:,var)]; % add constant feature to make array
    %metafeature_array = [repmat(ones(size(Metafeature)),1,1) Metafeature]; %% or use actual metafeature
    
    % check correlations of our simulated and actual metafeatures
    for j = 1:5; pearson_error_metafeature_simulated(var,j) = corr(squared_error(:,j),Metafeature_simu(non_nan_idx,j,var),'type','pearson'); end
    %[pearson_error_metafeature pearson_error_metafeature_simulated];
   
    %%%%%%%%%% METAFEATURE ARRAY SIMULATION END %%%%%%%%%%%%%%
 
    % Make predictions
    [pred_ST(non_nan_idx,:,var),pred_stack_ls,pred_stack_rdg,pred_FWLS_ls,pred_FWLS_rdg] = predictPhenotype_mf_simulation_V2(Predictions,vars_target,metafeature_array);
    pred_stack_all(non_nan_idx,:,var) = [pred_stack_ls pred_stack_rdg pred_FWLS_ls pred_FWLS_rdg];
    
    % Note prediction accuracies
    [MSE_stack_ST(var,:),MSE_stack_all(var,:),EV_ST(var,:),EV_all(var,:)] = prediction_accuracy_stats(vars_target,pred_ST(non_nan_idx,:,var),pred_stack_all(non_nan_idx,:,var));
           

end
    
   
%% Plot simulation results
    
%%% EXPLAINED VARIANCE
% (1) LSQ (2) Ridge (3) FWLS LSQ (4) FWLS Ridge
figure()
scatter(1:n_var,EV_all(:,2),'o','g'); hold on
scatter(1:n_var,EV_all(:,4),'o','k')
for rep = 1:n_repetitions
    scatter(1:n_var,EV_ST(:,rep),'x','b')
end
legend('Stacked Ridge','FWLS Ridge','HMM reps')
title(sprintf('Explained variance, simulated \\rho = %1.2f', p_metafeature(1)))
xlabel('Variable'); ylabel('Explained Variance');

%%% MEAN PREDICTION ERROR
% (1) LSQ (2) Ridge (3) FWLS LSQ (4) FWLS Ridge
figure()
scatter(1:n_var,MSE_stack_all(:,2),'o','g'); hold on
scatter(1:n_var,MSE_stack_all(:,4),'o','k')
for rep = 1:5
    scatter(1:n_var,MSE_stack_ST,'x','b')
end
legend('Stacked Ridge','FWLS Ridge','HMM reps')
title(sprintf('Mean prediction error, \\rho = %1.2f', p_metafeature(1)))
xlabel('Variable'); ylabel('Mean Prediction Error');

%%
% check the distribution of the simulated metafeature vs the actual
% metafeature (after being Gaussianized)
% Note: the simulated metafeature was given a Gaussian distribution
Metafeature_gauss = Likelihood_gaussianized(:,[2 6 14 25 44]);
for rep = 1:5
    figure()
    subplot(3,1,1)
    histogram(Metafeature(:,rep))
    xlabel('Metafeature'); ylabel('Frequency');
    subplot(3,1,2)
    histogram(Metafeature_gauss(:,rep))
    xlabel('Metafeature Gaussianized'); ylabel('Frequency');
    subplot(3,1,3)
    histogram(Metafeature_simu(:,rep))
    xlabel('Metafeature (Simulated)'); ylabel('Frequency');
end


%% Plot scatter graphs of metafeature data to look at distribution
for var = 1:34
    
    % remove NaNs
    [vars_target,Predictions,Metafeature,squared_error_gaussianized,non_nan_idx] = nan_subject_remove(var,vars_norm,vars_hat_norm,Metafeature_all,Squared_error_gaussianized);
    
    % Note squared error
    squared_error = (vars_target - Predictions).^2;
    
    % Plot relationships between error and simulated metafeature
    figure(1)
    for rep = 1%:5
        mf = Metafeature_simu(non_nan_idx,rep,var);
        [pearson_corr,pval_pearson] = corr(mf,squared_error(:,rep));
        [spearman_corr,pval_spearman] = corr(mf,squared_error(:,rep),'Type','Spearman');
        subplot(6,6,var)
        scatter(mf,squared_error(:,rep)); hold on
    end
    xlabel('Simulated metafeature (entropy)') ;ylabel('Prediction error')
    sgtitle(sprintf('SIMULATION - metafeature value vs squared error per subject (\\rho = %.2f)',p_metafeature))
    title(sprintf('Var #%d, P %.2f, S %.2f', var, pearson_corr, spearman_corr));
    if pval_pearson<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f', var, pearson_corr, spearman_corr)); end; if pval_spearman<0.05; title(sprintf('Var #%d, P %.2f, S %.2f*', var, pearson_corr, spearman_corr)); end; if pval_pearson<0.05 && pval_spearman<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f*', var, pearson_corr, spearman_corr)); end;
    
    
    
    % Plot relationships between error and actual metafeature
    figure(2)
    for rep = 1%:5
        mf = Metafeature_all(non_nan_idx,rep);
        sq_er = squared_error(:,rep);
        [pearson_corr,pval_pearson] = corr(mf,sq_er);
        [spearman_corr,pval_spearman] = corr(mf,sq_er,'Type','Spearman');
        subplot(6,6,var)
        scatter(mf,sq_er);
        hold on
    end
    
    xlabel('Likelihood'); ylabel('Prediction error')
    sgtitle(sprintf('Metafeature value vs squared error per subject (for a single repetition of HMM), P = Pearson, S = Spearman, * = Significant'))
    title(sprintf('Var #%d, P %.2f, S %.2f', var, pearson_corr, spearman_corr));
    if pval_pearson<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f', var, pearson_corr, spearman_corr)); end
    if pval_spearman<0.05; title(sprintf('Var #%d, P %.2f, S %.2f*', var, pearson_corr, spearman_corr)); end
    if pval_pearson<0.05 && pval_spearman<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f*', var, pearson_corr, spearman_corr)); end
    
end




%%
% To do
% Then: mess about with p and method for obtaining stacking weights DONE
% Then: mess about with metafeatures simulation to see what works and what
% doesn't, e.g.
% - introduce a new metafeature 0.1 correlation doesn't do much for 1 metafeature, 
% but what about 2 metafeatures at 0.1 each? DONE
% - can we introduce a nonlinear relationship and still see improvements? Might need a nonlinear method


% n_outliers_vars = [(1:34)' NaN(34,1)];
% n_unique_vars = [(1:34)' NaN(34,1)];
% n_nan_vars = [(1:34)' NaN(34,1)];
% for i = 1:34; n_unique_vars(i,2) = sum(~isnan((unique(vars(:,i))))); end % check to see if discrete vs continuous (vs continuous but repeated values common)
% for i = 1:34; n_nan_vars(i,2) = sum(isnan(vars(:,i))); end % check NaNs in data
% for i = 1:34; n_outliers_vars(i,2) = sum(isoutlier(vars(:,i))); end % check outliers in data


function meta_simu = metafeature_simulation_creation(pred_acc,p_meta,Metafeature)
    n_subjects = size(pred_acc,1);
    v_meta = randn(n_subjects, 1); % create random variable 'v'
    u_meta = (pred_acc ./ std(pred_acc)) - mean(pred_acc); % reverse: x = (s1 * u + m1)' to get u = randn(1, n);
    y_meta = std(Metafeature) .* squeeze((p_meta .* u_meta + sqrt(1 - p_meta.^2) .* v_meta));
    y_meta = y_meta + mean(Metafeature) - mean(y_meta); % set mean of rv to desired mean (Update: this doesn't make a difference)
    meta_simu = y_meta;
end



function [vars_target_clean,Predictions_clean,Metafeature_clean, squared_error_gaussianized_clean,non_nan_idx] = ...
                        nan_subject_remove(var,vars,vars_hat,Metafeature_all,Squared_error_gaussianized)

    % Store data for variable
    Predictions_clean = squeeze(vars_hat(:,var,:));
    vars_target_clean = vars(:,var);
    Metafeature_clean = Metafeature_all;
    squared_error_gaussianized_clean = Squared_error_gaussianized(:,:,var);
    
    % Remove subjects with missing values
    non_nan_idx = ~isnan(vars_target_clean);
    which_nan = isnan(vars_target_clean);
    if any(which_nan)
        vars_target_clean = vars_target_clean(~which_nan);
        Predictions_clean = Predictions_clean(~which_nan,:);
        Metafeature_clean = Metafeature_clean(~which_nan,:);
        squared_error_gaussianized_clean = squared_error_gaussianized_clean(~which_nan,:);
        warning('NaN found on Yin, will remove...')
    end
    
end

function [vars_target_clean,Predictions_clean,Metafeature_clean,squared_error_gaussianized_clean,non_outlier_idx] = ...
                        outlier_remove(vars_target,Predictions,Metafeature,squared_error_gaussianized)
                    
    vars_target_clean = vars_target;
    Predictions_clean = Predictions;
    Metafeature_clean = Metafeature;
    squared_error_gaussianized_clean = squared_error_gaussianized;
                    
    % Note index of outliers
    non_outlier_idx = find(~isoutlier(vars_target));
    which_outlier = isoutlier(vars_target);
    if any(which_outlier)
        vars_target_clean = vars_target(~which_outlier);
        Predictions_clean = Predictions(~which_outlier,:);
        Metafeature_clean = Metafeature(~which_outlier,:);
        squared_error_gaussianized_clean = squared_error_gaussianized(~which_outlier,:);
        warning('Outliers found on Yin, will remove...')
    end
                    
end

   
