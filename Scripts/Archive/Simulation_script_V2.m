%% SIMULATING METAFEATURES
clear; clc; %close all;
rng('default') % set seed for reproducibility
%DirOut = '/Users/bengriffin/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps/'; % Set directory
DirOut = '/Users/au699373/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_stack_ALL_vs_V4/'; % Test folder
%DirOut = '/Users/au699373/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/LOO_100_subjects/'; % Test folder

%figure(); image(imread([DirOut 'rho_stacked_accuracy_EV.jpg']));
%figure(); image(imread([DirOut 'rho_stacked_accuracy_MSE.jpg']));

% load data
%data_struct = struct()
%data_struct.preds = load([ DirOut  'HMM_predictions.mat']) % load predictions
load([ DirOut  'HMM_predictions.mat'])%load([ DirOut  'HMM_predictions.mat']) % load predictions
load('/Users/au699373/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/vars_target.mat') % load targets
load([ DirOut 'HMMs_meta_data_subject']); %load([ DirOut 'HMMs_meta_data_GROUP']); % load metadata
load([ DirOut 'HMMs_meta_data_subject_gaussianized']) % load Gaussianized metadata
%[Entropy_gaussianized,Likelihood_gaussianized] = gaussianize_r(); % Gaussianize metafeatures (using script in R) (note: must change the save directory to the correct folder in R)


% Select simulation choices
simulation_options.corr_type = 'Pearson'; % choose type of correlation
simulation_options.p_simulate = 0; % Add artifical correlation to simulation?
simulation_options.rep_selection = 1:5;%[2 6 14 25 44];  Select repetitions to use for simulation
simulation_options.n_folds = 10;
n_metafeatures = 2; % inc constant metafeature

% select metafeature e.g. Entropy, Entropy (gaussianized), Likelihood, Likelihood (gaussianized)
Metafeature_all = likelihood_subject; %Metafeature_all = Entropy_subject;
Metafeature_gaussianized = Likelihood_gaussianized;


% Note dimensions of arrays
n_subjects = size(vars_hat,1);
n_var = size(vars_hat,2);
n_repetitions = size(simulation_options.rep_selection,2);



% Store selections
vars_hat = vars_hat(:,:,simulation_options.rep_selection);
Metafeature_all = Metafeature_all(:,simulation_options.rep_selection);
Metafeature_gauss = Metafeature_gaussianized(:,simulation_options.rep_selection);



% normalize variables and target (should we remove outliers before normalizing since this will change max/min?
%  if we normalize each prediction, then we stack the predictions,
% how do we de-normalize? since we change the predictions by different
% amounts
vars_norm =  (vars-min(vars))./(max(vars)-min(vars));
vars_hat_norm = (vars_hat-min(vars_hat))./(max(vars_hat)-min(vars_hat));
Metafeature_norm = (Metafeature_all-min(Metafeature_all))./(max(Metafeature_all)-min(Metafeature_all)); % (Update: this does nothing - the weights just basically become very very small. If the metafeature values were super small, then this might matter because they is an upper limit on the weights)
Metafeature_gauss_norm = (Metafeature_gauss-min(Metafeature_gauss))./(max(Metafeature_gauss)-min(Metafeature_gauss));

% Question: why does dividing by std work so well?
% vars_norm =  vars./nanstd(vars);
% vars_hat_norm = vars_hat./nanstd(vars_hat);
% Metafeature_norm = Metafeature_all./nanstd(Metafeature_all);

% Question: why does normalize() fct not work well?
% normalize() is actually standardization (mean 0 std 1)
%vars_norm = normalize(vars);
%vars_hat_norm = normalize(vars_hat);


% Intialise variables
EV_ST = NaN(n_var,n_repetitions,2); EV_all = NaN(n_var,4,2);
MSE_stack_ST = NaN(n_var,n_repetitions,2); MSE_stack_all = NaN(n_var,4,2);
pred_ST = NaN(n_subjects,n_var,n_repetitions,2); pred_stack_all = NaN(1001,n_var,4,2);
W_stack_all = NaN(n_repetitions*n_metafeatures,simulation_options.n_folds,4,n_var,2);
Metafeature_simu = NaN(n_subjects,n_repetitions,n_var);


for v = 1:34

    % remove NaNs
    [vars_target,Predictions,Metafeature,non_nan_idx] = nan_subject_remove(v,vars_norm,vars_hat_norm,Metafeature_norm);%,Squared_error_gaussianized);

    % remove outliers
    %[vars_target,Predictions,Metafeature,squared_error_gaussianized] = outlier_remove(vars_target,Predictions,Metafeature,squared_error_gaussianized);

    % determine accuracy of predictions (we want our metafeature to be correlated with this)
    squared_error = (vars_target - Predictions).^2;
    
    % simulate metafeature (to have same correlation as our actual metafeature + p_simulate)
    %Metafeature_simu(non_nan_idx,:,v) = metafeature_simulation_creation(squared_error,Metafeature,p_simulate,simulation_options);
    Metafeature_simu(non_nan_idx,:,v) = metafeature_simulation_creation(squared_error,Metafeature,simulation_options);
    
%     % set up metafeature array (note we add a 'constant' feature here
%     if strcmp(simulation_options.metafeature_choice , 'Simulate')
%         metafeature_array = [repmat(ones(size(Metafeature)),1,1) Metafeature_simu(non_nan_idx,:,var)];
%     elseif strcmp(simulation_options.metafeature_choice , 'True')
%         metafeature_array = [repmat(ones(size(Metafeature)),1,1) Metafeature];
%     end
 
    % Make predictions and store prediction accuracies (simulated mf)
    metafeature_array_sim = [repmat(ones(size(Metafeature)),1,1) Metafeature_simu(non_nan_idx,:,v)];
    [pred_ST(non_nan_idx,v,:,1),pred_stack_ls,pred_stack_rdg,pred_FWLS_ls,pred_FWLS_rdg,W_stack_ls,W_stack_ridge,W_FWLS_ls,W_FWLS_ridge] = predictPhenotype_mf_simulation_V2(Predictions,vars_target,metafeature_array_sim);
    pred_stack_all(non_nan_idx,v,:,1) = [pred_stack_ls pred_stack_rdg pred_FWLS_ls pred_FWLS_rdg]; W_stack_all(:,:,:,v,1) = [cat(3,W_stack_ls,W_stack_ridge,W_FWLS_ls,W_FWLS_ridge)];
    [MSE_stack_ST(v,:,1),MSE_stack_all(v,:,1),EV_ST(v,:,1),EV_all(v,:,1)] = prediction_accuracy_stats(vars_target,squeeze(pred_ST(non_nan_idx,v,:,1)),pred_stack_all(non_nan_idx,v,:,1));
    
    % Make predictions and store prediction accuracies (true mf)
    metafeature_array_real = [repmat(ones(size(Metafeature)),1,1) Metafeature];
    [pred_ST(non_nan_idx,v,:,2),pred_stack_ls,pred_stack_rdg,pred_FWLS_ls,pred_FWLS_rdg,W_stack_ls,W_stack_ridge,W_FWLS_ls,W_FWLS_ridge] = predictPhenotype_mf_simulation_V2(Predictions,vars_target,metafeature_array_real);
    pred_stack_all(non_nan_idx,v,:,2) = [pred_stack_ls pred_stack_rdg pred_FWLS_ls pred_FWLS_rdg]; W_stack_all(:,:,:,v,2) = [cat(3,W_stack_ls,W_stack_ridge,W_FWLS_ls,W_FWLS_ridge)];
    [MSE_stack_ST(v,:,2),MSE_stack_all(v,:,2),EV_ST(v,:,2),EV_all(v,:,2)] = prediction_accuracy_stats(vars_target,squeeze(pred_ST(non_nan_idx,v,:,2)),pred_stack_all(non_nan_idx,v,:,2));
           

end
    
   
% Plot simulation results
    
%%% EXPLAINED VARIANCE
% 2-dim: (1) LSQ (2) Ridge (3) FWLS LSQ (4) FWLS Ridge
% 3-dim: (1) Simulated (2) True metafeature
figure()
scatter(1:n_var,EV_all(:,2,1),'o','g'); hold on % LSQ (no mf)
scatter(1:n_var,EV_all(:,4,1),'o','k') % Ridge (simulated mf)
scatter(1:n_var,EV_all(:,4,2),'o','r') % Ridge (true mf)
% scatter(1:n_var,EV_all(:,3,1),'o','c') % LSQ NO CONSTRAINTS (simulated mf) % LSQ no constraints is just as good as ridge here
% scatter(1:n_var,EV_all(:,3,2),'o','y') % LSQ NO CONSTRAINTS  (true mf)
for rep = 1:n_repetitions
    scatter(1:n_var,EV_ST(:,rep,1),'x','b')
end
legend('LSQLIN (no mf)','FWLS Ridge (sim)','FWLS Ridge (true)','HMM reps')
title(sprintf('Explained variance'));%, simulated \\rho = %1.2f', p_metafeature(1)))
xlabel('Variable'); ylabel('Explained Variance');
%%
%%% MEAN PREDICTION ERROR
% 2-dim: (1) LSQ (2) Ridge (3) FWLS LSQ (4) FWLS Ridge
% 3-dim: (1) Simulated (2) True metafeature
figure()
scatter(1:n_var,MSE_stack_all(:,2,1),'o','g'); hold on
scatter(1:n_var,MSE_stack_all(:,4,1),'o','k')
scatter(1:n_var,MSE_stack_all(:,4,2),'o','r')
%scatter(1:n_var,MSE_stack_all(:,3,2),'o','c')
for rep = 1:n_repetitions
    scatter(1:n_var,MSE_stack_ST(:,rep,1),'x','b')
end
legend('LSQLIN (no mf)','FWLS Ridge (sim)','FWLS Ridge (true)','HMM reps')
title(sprintf('Mean prediction error'));%, \\rho = %1.2f', p_metafeature(1)))
xlabel('Variable'); ylabel('Mean Prediction Error');


%% Distribution of real vs simulated metafeature
% check the distribution of the simulated metafeature vs the actual
% metafeature (after being Gaussianized)
% Note: the simulated metafeature was given a Gaussian distribution
n_bins = 20;
for rep = 1%:5
    figure()
    sgtitle(sprintf('Distribution of metafeature values (real + simulated)'))
    subplot(3,1,1)
    histogram(Metafeature_norm(:,rep),n_bins)
    xlabel('Metafeature'); ylabel('Frequency');
    subplot(3,1,2)
    histogram(Metafeature_gauss_norm(:,rep),n_bins)
    xlabel('Metafeature Gaussianized'); ylabel('Frequency');
    subplot(3,1,3)
    histogram(Metafeature_simu(:,rep),n_bins)
    xlabel('Metafeature (Simulated)'); ylabel('Frequency');
    
end



%% Plot scatter graphs of metafeature data to look at distribution
%%% REMEMBER, THE ACTUAL METAFEATURE IS THE SAME USED FOR ALL VARIABLES, IT
%%% IS JUST THE ERROR CHANGING BASED ON THE ASSOCIATED PREDICTIONS FOR THAT
%%% PARTICULAR VARIABLES. HOWEVER, FOR THE SIMULATED METAFEATURE, SINCE WE
%%% MADE SURE IT WAS CORRELATED WITH THE ERROR FOR EACH VARIABLE, THE
%%% SIMUALTED METAFEATURE IS DIFFERENT FOR EACH VARIABLE
% for v = 1:34
%     
%     % remove NaNs
%     [vars_target,Predictions,Metafeature,non_nan_idx] = nan_subject_remove(v,vars_norm,vars_hat_norm,Metafeature_all);
%     
%     % Note squared error
%     squared_error = (vars_target - Predictions).^2;
%     
%     % Plot relationships between error and simulated metafeature
%     figure(1)
%     for rep = 1%:5
%         mf = Metafeature_simu(non_nan_idx,rep,v);
%         [pearson_corr,pval_pearson] = corr(mf,squared_error(:,rep),'Type','Pearson');
%         [spearman_corr,pval_spearman] = corr(mf,squared_error(:,rep),'Type','Spearman');
%         subplot(6,6,v)
%         scatter(mf,squared_error(:,rep)); hold on
%     end
%     xlabel('Simulated metafeature (entropy)') ;ylabel('Prediction error')
%     sgtitle(sprintf('SIMULATION - metafeature value vs squared error per subject'))% (\\rho = %.2f)',p_metafeature))
%     title(sprintf('Var #%d, P %.2f, S %.2f', v, pearson_corr, spearman_corr));
%     if pval_pearson<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f', v, pearson_corr, spearman_corr)); end; if pval_spearman<0.05; title(sprintf('Var #%d, P %.2f, S %.2f*', v, pearson_corr, spearman_corr)); end; if pval_pearson<0.05 && pval_spearman<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f*', v, pearson_corr, spearman_corr)); end;
%    
%     
%     % Plot relationships between error and actual metafeature
%     figure(2)
%     for rep = 1%:5
%         mf = Metafeature_all(non_nan_idx,rep);
%         sq_er = squared_error(:,rep);
%         [pearson_corr,pval_pearson] = corr(mf,sq_er,'Type','Pearson');
%         [spearman_corr,pval_spearman] = corr(mf,sq_er,'Type','Spearman');
%         subplot(6,6,v)
%         scatter(mf,sq_er);
%         hold on
%     end
%     
%     xlabel('Likelihood'); ylabel('Prediction error')
%     sgtitle(sprintf('Metafeature value vs squared error per subject (for a single repetition of HMM), P = Pearson, S = Spearman, * = Significant'))
%     title(sprintf('Var #%d, P %.2f, S %.2f', v, pearson_corr, spearman_corr));
%     if pval_pearson<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f', v, pearson_corr, spearman_corr)); end
%     if pval_spearman<0.05; title(sprintf('Var #%d, P %.2f, S %.2f*', v, pearson_corr, spearman_corr)); end
%     if pval_pearson<0.05 && pval_spearman<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f*', v, pearson_corr, spearman_corr)); end
% 
%     figure(3)
%     for rep = 1%:5
%         mf_sim = Metafeature_simu(non_nan_idx,rep,v);
%         mf_real = Metafeature_all(non_nan_idx,rep);
%         subplot(6,6,v)
%         scatter(mf_sim,mf_real); hold on
%         sgtitle('Simulated metafeature values vs actual metafeature value')
%     end
%     
% end

%% What's the difference between the actual metafeature and our simulated one?
n_subjects = size(Metafeature_norm,1);
% What's the difference between our simulated metafeature and the real one?
% boxplots for simulated vs actual metafeature
figure();
subplot(2,1,1); boxplot(Metafeature_norm); xlabel('HMM repetition');
title('Actual metafeature'); ylabel('Normalized metafeature value');
subplot(2,1,2); boxplot(Metafeature_simu(:,:,1)); xlabel('HMM repetition');
title('Simulated metafeature'); ylabel('Normalized metafeature value');

%% Plot heatmaps of correlations between errors of predictions and metafeatures
all_corr_error_metafeature = NaN(n_var,n_repetitions,4);
all_corr_error_metafeature_simulated = NaN(n_var,n_repetitions,4);
all_corr_error_metafeature_gauss = NaN(n_var,n_repetitions,4);
corr_measures = {'Pearson','Spearman','Kendall','Distance'};

for v = 1:n_var
    [vars_target,Predictions,Metafeature,non_nan_idx] = nan_subject_remove(v,vars_norm,vars_hat_norm,Metafeature_norm); % remove NaNs
    squared_error = (vars_target - Predictions).^2; % Note squared error
    Metafeature_simu(non_nan_idx,:,v) = metafeature_simulation_creation(squared_error,Metafeature,simulation_options);
    for k = 1:3
        corr_type = corr_measures{k};
        for j = 1:n_repetitions;[all_corr_error_metafeature(v,j,k),~] = corr(squared_error(:,j),Metafeature(:,j),'type',corr_type); end
        for j = 1:n_repetitions; [all_corr_error_metafeature_simulated(v,j,k),~] = corr(squared_error(:,j),Metafeature_simu(non_nan_idx,j,v),'type',corr_type); end
        for j = 1:n_repetitions; [all_corr_error_metafeature_gauss(v,j,k),~] = corr(squared_error(:,j),Metafeature_gaussianized(non_nan_idx,j),'type',corr_type); end
    end
%     % compare distance correlations of real and simulated metafeatures
%     for j = 1:n_repetitions; all_corr_error_metafeature(v,j,4) = distcorr(squared_error(:,j),Metafeature(:,j)); end
%     for j = 1:n_repetitions; all_corr_error_metafeature_simulated(v,j,4) = distcorr(squared_error(:,j),Metafeature_simu(non_nan_idx,j,v)); end
%     for j = 1:n_repetitions; all_corr_error_metafeature_gauss(v,j,4) = distcorr(squared_error(:,j),Metafeature_gaussianized(non_nan_idx,j)); end
end

for j = 1:3%4
    ax_min = [min(min(all_corr_error_metafeature(:,:,j))),min(min(all_corr_error_metafeature_simulated(:,:,j))),min(min(all_corr_error_metafeature_gauss(:,:,j)))];
    ax_max = [max(max(all_corr_error_metafeature(:,:,j))),max(max(all_corr_error_metafeature_simulated(:,:,j))),max(max(all_corr_error_metafeature_gauss(:,:,j)))];
    figure()
    sgtitle(sprintf('%s Correlation of prediction error vs metafeature',corr_measures{j}))
    subplot(1,3,1); imagesc(all_corr_error_metafeature(:,:,j)); colorbar; caxis([min(ax_min) max(ax_max)]);
    xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('True metafeature'))
    subplot(1,3,2); imagesc(all_corr_error_metafeature_gauss(:,:,j)); colorbar; caxis([min(ax_min) max(ax_max)]);
    xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Gaussianized metafeature'))  
    subplot(1,3,3); imagesc(all_corr_error_metafeature_simulated(:,:,j)); colorbar; caxis([min(ax_min) max(ax_max)]);
    xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Simulated metafeature'))

end
% 
% % Total correlation across all 34 variables for each repetition
% var_keep = [2:17 19 21:22 26 30:34]; % only keep non-terrible predictions
% % mean(abs(all_corr_error_metafeature(var_keep,:,1)))
% % Max FO : 0.0448    0.0486    0.0434    0.0380    0.0521
% % L'hood : 0.0386    0.0489    0.0286    0.0427    0.0419
% % Entropy: 0.0328    0.0512    0.0440    0.0317    0.0531
%%
pearson_mf_real  = all_corr_error_metafeature(:,:,1); % pearson
pearson_mf_sim  = all_corr_error_metafeature_simulated(:,:,1); % pearson
figure()
scatter(pearson_mf_real,pearson_mf_sim)
title('Pearson Correlation')
xlabel('True metafeature'); ylabel('Simulated metafeature');
legend('Repetition 1', 'Repetition 2', 'Repetition 3', 'Repetition 4', 'Repetition 5')

spearman_mf_real = all_corr_error_metafeature(:,:,2); % spearman
spearman_mf_sim  = all_corr_error_metafeature_simulated(:,:,2); % spearman
figure()
scatter(spearman_mf_real,spearman_mf_sim)
title('Spearman Correlation')
xlabel('True metafeature'); ylabel('Simulated metafeature');
legend('Repetition 1', 'Repetition 2', 'Repetition 3', 'Repetition 4', 'Repetition 5')

%%
for v = 1
mean_mf = [mean(Metafeature_norm); mean(Metafeature_simu(non_nan_idx,:,v))]
std_mf = [std(Metafeature_norm); std(Metafeature_simu(non_nan_idx,:,v))]
kurt_mf = [kurtosis(Metafeature_norm); kurtosis(Metafeature_simu(non_nan_idx,:,v))]
skew_mf = [skewness(Metafeature_norm); skewness(Metafeature_simu(non_nan_idx,:,v))]
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


function meta_simu = metafeature_simulation_creation(squared_error,Metafeature,simulation_options)
    n_repetitions = size(Metafeature,2);
    n_subjects = size(squared_error,1);
    meta_simu = NaN(n_subjects,n_repetitions);
    
    pearson_error_metafeature = NaN(1,n_repetitions);
    for rep = 1:n_repetitions
        [pearson_error_metafeature(rep), ~] = corr(squared_error(:,rep),Metafeature(:,rep),'type',simulation_options.corr_type);
    end
    p_meta = pearson_error_metafeature + simulation_options.p_simulate; % make correlation more favourable
    %p = 0.2;
    %p_meta = [p p p p p];

    theta = acos(p_meta); % corresponding angle
    x1 = squared_error; % generate fixed given variable (this will be squared error for me) % THIS IS WHAT WE WANT TO GENERAET A VECTOR SIMILAR TO
    x2 = randn(n_subjects,5); % generate rv variable that 
    Xctr1 = x1 - mean(x1); % center columns to mean 0
    Xctr2 = x2 - mean(x2);% center columns to mean 0
    Id = eye(n_subjects); % generate identity matrix size n_subjects x n_subjects

    for i = 1:5
        P = Xctr1(:,i)*inv(Xctr1(:,i)'*Xctr1(:,i)) * Xctr1(:,i)'; % get projection matrix onto space by defined by x1
        Xctr2_orth = (P-Id) * Xctr2(:,i); % make Xctr2 orthogonal to Xctr1 using projection matrix
        Y1 = Xctr1(:,i) * 1./(sqrt(sum(Xctr1(:,i).^2))); % scale columns to length 1
        Y2 = Xctr2_orth * 1./(sqrt(sum(Xctr2_orth.^2))); % scale columns to length 1
        meta_simu(:,i) = Y2 + (1/tan(theta(:,i))) * Y1; % final new vector
    end
    corr(squared_error,meta_simu)  % check this is desired correlation



end



function [vars_target_clean,Predictions_clean,Metafeature_clean,non_nan_idx] = ...
                        nan_subject_remove(var,vars,vars_hat,Metafeature_norm)%,Squared_error_gaussianized)

    % Store data for variable
    Predictions_clean = squeeze(vars_hat(:,var,:));
    vars_target_clean = vars(:,var);
    Metafeature_clean = Metafeature_norm;
    %squared_error_gaussianized_clean = Squared_error_gaussianized(:,:,var);
    
    % Remove subjects with missing values
    non_nan_idx = ~isnan(vars_target_clean);
    which_nan = isnan(vars_target_clean);
    if any(which_nan)
        vars_target_clean = vars_target_clean(~which_nan);
        Predictions_clean = Predictions_clean(~which_nan,:);
        Metafeature_clean = Metafeature_clean(~which_nan,:);
        %squared_error_gaussianized_clean = squared_error_gaussianized_clean(~which_nan,:);
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

   

% function [Entropy_gaussianized,Likelihood_gaussianized] = gaussianize_r()
%     setenv('PATH', getenv('PATH')+":/usr/local/bin")
% 
%     % Run R script
%     !Rscript gaussianize_metafeatures.R
%     %!Rscript Users/bengriffin/OneDrive\ -\ Aarhus\ Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR\ Results/FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps/gaussianize_metafeatures.R
%     %!Rscript  Users/bengriffin/OneDrive\ -\ Aarhus\ Universitet/Dokumenter/R/gaussianize_metafeatures.R
%     
%     % Load the data saved in the R script
%     gauss_struct = load([ DirOut 'HMMs_meta_data_subject_gaussianized.mat'],'Entropy_gaussianized','Likelihood_gaussianized');
%     Entropy_gaussianized = gauss_struct.Entropy_gaussianized;
%     Likelihood_gaussianized = gauss_struct.Likelihood_gaussianized;
% end



%p_metafeature = pearson_error_metafeature(var,:) + p_simulate; % make correlation more favourable


% figure()
% for i = 1%:n_repetitions
%     Means = [mean(Metafeature_all(:,i)); mean(Metafeature_gauss(:,i)); mean(Metafeature_simu(:,i,1))];
%     Stds = [std(Metafeature_all(:,i)); std(Metafeature_gauss(:,i)); std(Metafeature_simu(:,i,1))];
%     LowerQ = [quantile(Metafeature_all(:,i),0.25); quantile(Metafeature_gauss(:,i),0.25); quantile(Metafeature_simu(:,i,1),0.25)];
%     Medians = [median(Metafeature_all(:,i)); median(Metafeature_gauss(:,i)); median(Metafeature_simu(:,i,1))];
%     UpperQ = [quantile(Metafeature_all(:,i),0.75); quantile(Metafeature_gauss(:,i),0.75); quantile(Metafeature_simu(:,i,1),0.75)];
%     InterQR = [iqr(Metafeature_all(:,i)); iqr(Metafeature_gauss(:,i)); iqr(Metafeature_simu(:,i,1))];
%     
%     row_names = {'Actual Metafeature';'Actual Metafeature (Gauss)'; 'Simulated Metafeature'};
%     T = table(Means,Stds,LowerQ,Medians,UpperQ,InterQR,'RowNames',row_names);
%     
%     % Get the table in string form.
%     TString = evalc('disp(T)');
%     % Use TeX Markup for bold formatting and underscores.
%     TString = strrep(TString,'<strong>','\bf');
%     TString = strrep(TString,'</strong>','\rm');
%     TString = strrep(TString,'_','\_');
%     % Get a fixed-width font.
%     FixedWidth = get(0,'FixedWidthFontName');
%     % Output the table using the annotation command.
%     annotation(gcf,'Textbox','String',TString,'Interpreter','Tex',...
%         'FontName',FixedWidth,'Units','Normalized','Position',[0 0 1 1]);
% 
% end


% %%
% % let's plot them
% figure(); sgtitle('Metafeature (Simulated)')
% subplot(3,1,1); scatter(1:n_subjects,Metafeature_simu(:,1,1));
% subplot(3,1,2); scatter(1:n_subjects,sort(Metafeature_simu(:,1,1)))
% subplot(3,1,3); scatter(Metafeature_simu(:,1,1),squared_error(:,1))
% figure(); sgtitle('Metafeature (Actual)')
% subplot(3,1,1); scatter(1:n_subjects,Metafeature_all(:,1))
% subplot(3,1,2); scatter(1:n_subjects,sort(Metafeature_all(:,1)))
% subplot(3,1,3); scatter(Metafeature_all(:,1),squared_error(:,1))
% figure(); sgtitle('Metafeature (Actual - Gaussianized)')
% subplot(3,1,1); scatter(1:n_subjects,Metafeature_gauss(:,1))
% subplot(3,1,2); scatter(1:n_subjects,sort(Metafeature_gauss(:,1)))
% subplot(3,1,3); scatter(Metafeature_gauss(:,1),squared_error(:,1))


% % compare pearson's correlations of real and simulated metafeatures
% for j = 1:n_repetitions; [pearson_corr_error_metafeature(var,j), ~] = corr(squared_error(:,j),Metafeature(:,j),'type','pearson'); end
% for j = 1:n_repetitions; [pearson_corr_error_metafeature_simulated(var,j),~] = corr(squared_error(:,j),Metafeature_simu(non_nan_idx,j,var),'type','pearson'); end
% for j = 1:n_repetitions; [pearson_corr_error_metafeature_gauss(var,j),~] = corr(squared_error(:,j),Metafeature_gaussianized(non_nan_idx,j),'type','pearson'); end
% 
% 
% % compare spearman's correlations of real and simulated metafeatures
% for j = 1:n_repetitions; [spearman_corr_error_metafeature(var,j), ~] = corr(squared_error(:,j),Metafeature(:,j),'type','spearman'); end
% for j = 1:n_repetitions; [spearman_corr_error_metafeature_simulated(var,j),~] = corr(squared_error(:,j),Metafeature_simu(non_nan_idx,j,var),'type','spearman'); end
% for j = 1:n_repetitions; [spearman_corr_error_metafeature_gauss(var,j),~] = corr(squared_error(:,j),Metafeature_gaussianized(non_nan_idx,j),'type','spearman'); end
% 

% ax_min = [min(min(pearson_corr_error_metafeature)),min(min(pearson_corr_error_metafeature_simulated)),min(min(pearson_corr_error_metafeature_gauss))];
% ax_max = [max(max(pearson_corr_error_metafeature)),max(max(pearson_corr_error_metafeature_simulated)),max(max(pearson_corr_error_metafeature_gauss))];
% figure()
% subplot(1,3,1); imagesc(pearson_corr_error_metafeature); colorbar; caxis([min(ax_min) max(ax_max)]);
% xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Pearson corr of prediction err vs metafeature (real)'))
% subplot(1,3,2); imagesc(pearson_corr_error_metafeature_simulated); colorbar; caxis([min(ax_min) max(ax_max)]); 
% xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Pearson corr of prediction err vs metafeature (simulated)'))
% subplot(1,3,3); imagesc(pearson_corr_error_metafeature_gauss); colorbar; caxis([min(ax_min) max(ax_max)]); 
% xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Pearson corr of prediction err vs metafeature (real + Gaussianized)'))
% 
% ax_min = [min(min(spearman_corr_error_metafeature)),min(min(spearman_corr_error_metafeature_simulated)),min(min(spearman_corr_error_metafeature_gauss))];
% ax_max = [max(max(spearman_corr_error_metafeature)),max(max(spearman_corr_error_metafeature_simulated)),max(max(spearman_corr_error_metafeature_gauss))];
% figure()
% subplot(1,3,1); imagesc(spearman_corr_error_metafeature); colorbar; caxis([min(ax_min) max(ax_max)]);
% xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Spearman corr of prediction err vs metafeature (real)'))
% subplot(1,3,2); imagesc(spearman_corr_error_metafeature_simulated); colorbar; caxis([min(ax_min) max(ax_max)]);
% xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Spearman corr of prediction err vs metafeature (simulated)'))
% subplot(1,3,3); imagesc(spearman_corr_error_metafeature_gauss); colorbar; caxis([min(ax_min) max(ax_max)]); 
% xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Spearman corr of prediction err vs metafeature (real + Gaussianized)'))
% 
% ax_min = [min(min(dist_corr_error_metafeature)),min(min(dist_corr_error_metafeature_simulated)),min(min(dist_corr_error_metafeature_gauss))];
% ax_max = [max(max(dist_corr_error_metafeature)),max(max(dist_corr_error_metafeature_simulated)),max(max(dist_corr_error_metafeature_gauss))];
% figure()
% subplot(1,3,1); imagesc(dist_corr_error_metafeature); colorbar; caxis([min(ax_min) max(ax_max)]);
% xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Distance corr of prediction err vs metafeature (real)'))
% subplot(1,3,2); imagesc(dist_corr_error_metafeature_simulated); colorbar; caxis([min(ax_min) max(ax_max)]);
% xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Distance corr of prediction err vs metafeature (simulated)'))
% subplot(1,3,3); imagesc(dist_corr_error_metafeature_gauss); colorbar; caxis([min(ax_min) max(ax_max)]); 
% xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Distance corr of prediction err vs metafeature (real + Gaussianized)'))


% simulation_options.metafeature_choice = 'Simulate'; % 'True' % choose whether to simulated metafeature or use the true metafeature
%     % set up metafeature array (note we add a 'constant' feature here
%     if strcmp(simulation_options.metafeature_choice , 'Simulate')
%         metafeature_array = [repmat(ones(size(Metafeature)),1,1) Metafeature_simu(non_nan_idx,:,var)];
%     elseif strcmp(simulation_options.metafeature_choice , 'True')
%         metafeature_array = [repmat(ones(size(Metafeature)),1,1) Metafeature];
%     end


% %% Let's take a look at the actual predictions
% %vars_hat_new = PREDICTIONS.*vars_hat_max_min + min(vars_hat);
% denormalized_pred_ST = pred_ST(:,:,:,1).*(max(vars_hat)-min(vars_hat)) + min(vars_hat);
% denormalized_pred_stack_all = pred_stack_all(:,:,:,1).*(max(vars_hat)-min(vars_hat)) + min(vars_hat);
% 
% squeeze(pred_new(:,1,:))
% 
% %%
% clc
% vars_hat_minus_min = (vars_hat-min(vars_hat));
% vars_hat_max_min = (max(vars_hat)-min(vars_hat));
% vars_hat_norm_new = (vars_hat_minus_min)./(vars_hat_max_min);
% PREDICTIONS = vars_hat_norm_new;
% %vars_hat_rev_1 = ((vars_hat_norm_new).*(max(vars_hat)-min(vars_hat)))  min(vars_hat);
% 
% 
% vars_hat_new = PREDICTIONS.*vars_hat_max_min + min(vars_hat);
% 
% squeeze(vars_hat_new(:,1,:))
% 
% 
% vars_minus_min = (vars-min(vars));
% vars_max_min = (max(vars)-min(vars));
% vars_norm_new = (vars_minus_min)./(vars_max_min);
% 
% vars_rev_1 = vars_norm_new.*(vars_max_min);
% vars_new = vars_rev_1 + min(vars);


   

%     v_meta = v_meta/std(v_meta);
%     u_meta = (squared_error ./ std(squared_error)) - mean(squared_error); % reverse: x = (s1 * u + m1)' to get u = randn(1, n);
%     
%     % (var(u_meta) == var(v_meta)); % these two variables need to have the same variance
%     %corrcoef(u_meta,repmat(v_meta,1,5)); and they should be uncorrelated
% 
%     y_meta = squeeze((p_meta .* u_meta + sqrt(1 - p_meta.^2) .* v_meta));
%     %y_meta = std(Metafeature) .* y_meta; % set std of rv to desired std
%     %y_meta = y_meta + mean(Metafeature) - mean(y_meta); % set mean of rv to desired mean (Update: this doesn't make a difference)
%     %y_meta = (y_meta-min(y_meta))./(max(y_meta)-min(y_meta)); % normalize simulated metafeature
%     meta_simu = y_meta;
% 
%     p_meta
%     p = NaN(1,n_repetitions);
%     for rep = 1:n_repetitions
%         [p(rep), ~] = corr(squared_error(:,rep),meta_simu(:,rep),'type',simulation_options.corr_type);
%     end
%     p

