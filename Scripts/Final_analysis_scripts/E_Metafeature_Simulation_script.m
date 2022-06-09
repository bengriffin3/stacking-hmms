%% SIMULATING METAFEATURES
%%% NB: I think instead of the remove NaN function, you might be able to
%%% ignore NaNs (e.g. using nanmean or corr(A,B, 'rows','complete'))
clear; clc; %close all;
rng('default') % set seed for reproducibility
%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_4\';
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\Varying_states\';


% load data
load('vars_target.mat') % load targets
load([ DirOut 'HMMs_meta_data_subject']); % load metadata
load([ DirOut 'HMMs_meta_data_GROUP']); % load metadata
%load([ DirOut 'HMMs_meta_data_GROUP_FINAL']); % load metadata
load([ DirOut  'HMM_predictions.mat']) % load predictions
%load([ DirOut 'simulated_metafeatures_V2.mat']); % load simulated metafeatures

%vars_hat = reshape(vars_hat,[1001 34 84]);



% Select simulation choices
simulation_options.rep_selection = 1:15%size(vars_hat,3);%[2 6 14 25 44];  Select repetitions to use for simulation (default is all repetitions)
simulation_options.n_folds = 10;
simulation_options.corr_specify = []%0.05%[]; % specify correlation between simulated metafeature and accuracy of HMM prediction (if empty, we use same correlation as true metafeatures)
simulation_options.simulate_corr_with =  'prediction_accuracy'; % 'vars_targets';
simulation_options.pred_method = 'KRR'; % 'KRR' 'KNN'
simulation_options.vars_selection = 1:34;
simulation_options.simulated_metafeatures_in_r = []%simulated_metafeatures(:,:,simulation_options.vars_selection,4); % leave empty (i.e. = []) to simulate metafeature in MATLAB

% add an option here where you can choose to include static predictions in the stacking (the corresponding metafeature would be all 1s I guess


% select metafeature (e.g. Entropy/Likelihood) % metastate PCs is wrong
% currently (not saving for each rep)
%metafeature_store = {metastate_profile likelihood_subject Entropy_subject_ln  Entropy_subject_log10 Entropy_subject_log2 maxFO_all_reps FO_metastate_new_1 FO_metastate_new_2 metastate_profile subject_distance_sum subject_distance_mean };% FO_metastate_new_1_gauss metastate_profile_gauss};
metafeature_store = {FO_PCs_aligned metastate_profile likelihood_subject Entropy_subject_log2 maxFO_all_reps FO_metastate_new_1 FO_metastate_new_2}; % FO_metastate_new_1_gauss metastate_profile_gauss};
%metafeature_store = {FO_PCs_aligned  likelihood_subject Entropy_subject_log2 maxFO_all_reps };% FO_metastate_new_1_gauss metastate_profile_gauss};
metafeature = [squeeze(metafeature_store{2}(:,simulation_options.rep_selection))];
%metafeature = ones(1001,15);

% if strcmp(simulation_options.pred_method,'KRR')
%     load([ DirOut  'HMM_predictions_KRR.mat']) % load predictions
%     explained_variance = explained_variance_KRR; vars_hat = vars_hat_KRR;
% elseif strcmp(simulation_options.pred_method,'KNN')
%     load([ DirOut  'HMM_predictions_KNN.mat']) % load predictions
%     explained_variance = explained_variance_KNN; vars_hat = vars_hat_KNN;
% else % use both methods
%     load([ DirOut  'HMM_predictions.mat']) % load predictions
%     metafeature = [metafeature metafeature];
% end


vars = vars(:,simulation_options.vars_selection);
vars_hat = vars_hat(:,simulation_options.vars_selection,:);


metafeature_norm = (metafeature-min(metafeature))./(max(metafeature)-min(metafeature));
vars_norm =  (vars-min(vars))./(max(vars)-min(vars));
vars_hat_norm = (vars_hat-min(vars_hat))./(max(vars_hat)-min(vars_hat));


% Run simulation
[pred_ST,pred_stack_all,MSE_stack_ST,MSE_stack_all,EV_ST,EV_all,~,metafeature_simu,~,squared_error] = simulation_full(vars,vars_hat,metafeature,simulation_options);


% Plot simulation results
n_var = size(vars_hat,2);
n_repetitions = length(simulation_options.rep_selection);
%total_EVs = [sum(EV_all(:,2,1)) sum(EV_all(:,4,1)) sum(EV_all(:,4,2)) sum(EV_all(:,5,1))];
total_EVs = [sum(EV_all(:,2,1)) sum(EV_all(:,4,1)) sum(EV_all(:,4,2))];
%%% EXPLAINED VARIANCE
% 2-dim: (1) LSQ (2) Ridge (3) FWLS LSQ (4) FWLS Ridge (5) Random Forest
% 3-dim: (1) Simulated (2) True metafeature
figure()
scatter(1:n_var,EV_all(:,2,1),'o','g'); hold on % LSQ (no mf)
scatter(1:n_var,EV_all(:,4,1),'o','k') % Ridge (simulated mf)
scatter(1:n_var,EV_all(:,4,2),'o','r') % Ridge (true mf)
% scatter(1:n_var,EV_all(:,3,1),'o','c') % LSQ NO CONSTRAINTS (simulated mf) % LSQ no constraints is just as good as ridge here
% scatter(1:n_var,EV_all(:,3,2),'o','y') % LSQ NO CONSTRAINTS  (true mf)
scatter(1:n_var,EV_all(:,5,1),'o','m') % Random forest (no mf)
for rep = 1:n_repetitions
    scatter(1:n_var,EV_ST(:,rep,1),'x','b')
end
%legend('LSQLIN (no mf)','FWLS Ridge (sim)','FWLS Ridge (true)','HMM reps')
%legend('LSQLIN (no mf)','FWLS Ridge (sim)','FWLS Ridge (true)','Random Forest', 'HMM reps')
legend('LSQLIN (no mf)','FWLS Ridge (sim)','FWLS Ridge (true)', 'HMM reps')
%title(sprintf('Explained variance, no mf = %.2f, sim = %.2f, true = %.2f, RF = %.2f' ,round(total_EVs,2)));%, simulated \\rho = %1.2f', p_metafeature(1)))
title(sprintf('Explained variance, no mf = %.2f, sim = %.2f, true = %.2f' ,round(total_EVs,2)));%, simulated \\rho = %1.2f', p_metafeature(1)))
xlabel('Variable'); ylabel('Explained Variance');


%% MEAN PREDICTION ERROR
%%% All repetitions
% 2-dim: (1) LSQ (2) Ridge (3) FWLS LSQ (4) FWLS Ridge
% 3-dim: (1) Simulated (2) True metafeature
total_MSEs = [sum(MSE_stack_all(:,2,1)) sum(MSE_stack_all(:,4,1)) sum(MSE_stack_all(:,4,2)) sum(MSE_stack_all(:,5,1))];
figure()
scatter(1:n_var,MSE_stack_all(:,2,1),'o','g'); hold on
scatter(1:n_var,MSE_stack_all(:,4,1),'o','k')
scatter(1:n_var,MSE_stack_all(:,4,2),'o','r')
%scatter(1:n_var,MSE_stack_all(:,3,2),'o','c')
scatter(1:n_var,MSE_stack_all(:,5,1),'o','m') % Random forest (no mf)
for rep = 1:n_repetitions
    scatter(1:n_var,MSE_stack_ST(:,rep,1),'x','b')
end
legend('LSQLIN (no mf)','FWLS Ridge (sim)','FWLS Ridge (true)','HMM reps')
title(sprintf('Mean prediction error, no mf = %.2f, sim = %.2f, true = %.2f, RF = %.2f',round(total_MSEs,2)));%, simulated \\rho = %1.2f', p_metafeature(1)))
xlabel('Variable'); ylabel('Mean Prediction Error');

%% MEAN PREDICTION ERROR
%%% Best repetition only
% 2-dim: (1) LSQ (2) Ridge (3) FWLS LSQ (4) FWLS Ridge
% 3-dim: (1) Simulated (2) True metafeature
figure()
scatter(1:n_var,MSE_stack_all(:,2,1),'o','g'); hold on
scatter(1:n_var,MSE_stack_all(:,4,1),'o','k')
scatter(1:n_var,MSE_stack_all(:,4,2),'o','r')
%scatter(1:n_var,MSE_stack_all(:,3,2),'o','c')
scatter(1:n_var,MSE_stack_all(:,5,1),'o','m') % Random forest (no mf)
scatter(1:n_var,min(MSE_stack_ST(:,:,1)')','x','b')
legend('LSQLIN (no mf)','FWLS Ridge (sim)','FWLS Ridge (true)', 'Stacked Random Forest', 'Best HMM rep')
title(sprintf('Mean prediction error, no mf = %.2f, sim = %.2f, true = %.2f, RF = %.2f',round(total_MSEs,2)));%, simulated \\rho = %1.2f', p_metafeature(1)))
xlabel('Variable'); ylabel('Mean Prediction Error');

 %% Correlation between PCs of FO matrices of distinct repetitions of HMM
figure;
for pc = 1:6
    subplot(2,3,pc);
    imagesc(corr(squeeze(FO_PCs_aligned(:,pc,:))))
    title(sprintf('PC %i', pc))
    colorbar;
end
sgtitle('Correlation between PCs of FO matrices of distinct repetitions of HMM')
%% Correlation between PCs of FO matrices and accuracies of HMM repetition
% Goal is to have for each repetition, and each variable, the accuracy of
% that repetition (i.e. squared error) and then check the correlation
% between this squared error and the PCs

FO_PCs_aligned;
p_store = NaN(34,15,6);
figure
for pc = 1:6
for v = 1:n_var
    [~,~,~,non_nan_idx] = nan_subject_remove(v,vars_norm,vars_hat_norm,metafeature_norm); % remove NaNs (this needs to be done for every var because different subjects are missing for each var)
    for j = 1:n_repetitions
        p_store(v,j,pc) = corr(squared_error(non_nan_idx,j,v),FO_PCs_aligned(non_nan_idx,pc,j),'Rows','complete');
    end

end
subplot(2,3,pc)
imagesc(p_store(:,:,pc)); colorbar
title(sprintf('Principal Component %i',pc))
xlabel('Accuracy of HMM Repetition'); ylabel('Intelligence Variable')
end
sgtitle('Correlation between PCs of FO matrices and accuracies of HMM repetitions')


% figure()
% sgtitle(sprintf('%s Correlation of prediction error vs metafeature',corr_measures{j}))
% subplot(1,2,1); imagesc(all_corr_error_metafeature(:,:,j)); colorbar; caxis([min(ax_min) max(ax_max)]);
% xlabel('HMM repetition (increasing K->)'); ylabel('Variable'); title(sprintf('True metafeature'))  
 




%% Plot histogram of actual variables
figure;
% let's plot for first variable for now
for i = 1:34
    subplot(6,6,i)
    histogram(vars(:,i))
    title(sprintf('Variable no. %i',i))
end
sgtitle('Distribution of target features for 34 intelligence variables')


%% Plot histogram of predictions
figure;
% let's plot for first variable for now
var_plot = squeeze(vars_hat(:,1,:));
for i = 1:15
    subplot(5,3,i)
    histogram(var_plot(:,i))
    title(sprintf('HMM repetition %i', i))
end
sgtitle('Predictions for variable 1')

%% Plot histograms of predictions as well as actual variables (non-stacked predictions)
% Let's just show the first prediction of each one
% let's plot for first variable for now
figure;
for i = 1:34
    subplot(6,6,i)
    histogram(vars(:,i)); hold on
    var_plot = squeeze(vars_hat(:,i,:));
    histogram(var_plot(:,2))

    title(sprintf('Variable no. %i',i))
end
sgtitle('Distribution of target features and predictions for 34 intelligence variables')

%% Plot histograms of predictions as well as actual variables
figure;
for i = 1:34
    subplot(6,6,i)
    histogram(vars_norm(:,i)); hold on
    %histogram(squeeze(pred_stack_all(:,i,5,1))) %%% STACKED RANDOM FORESTS
    histogram(squeeze(pred_stack_all(:,i,2,1))) %%% STACKED RIDGE REGRESSION
    title(sprintf('Variable no. %i',i))
end
sgtitle('Distribution of target features and predictions for 34 intelligence variables')


%% Plot histogram of errors of predictions
for v = 1
figure;
for i = 1:15
    subplot(5,3,i)
    % we just plot first variable for now
    histogram(squared_error(:,i,v))
    title(sprintf('HMM repetition %i', i))
end
sgtitle(sprintf('Error of predictions for variable %i', v))
end



%% Correlation between metafeatures of different repetitions
figure;
subplot(1,2,1); imagesc(corr(metafeature,'rows','complete')); colorbar; xlabel('Metafeature of HMM repetition'); ylabel('Metafeature of HMM repetition'); title('Real metafeature')
subplot(1,2,2); imagesc(corr(metafeature_simu(:,:,22),'rows','complete')); colorbar; xlabel('Metafeature of HMM repetition'); ylabel('Metafeature of HMM repetition'); title('Simulated metafeature')
sgtitle(sprintf('Matrices of correlations between metafeatures (real and simulated) of distinct HMM repetitions'))

%% Correlation between metafeatures of different repetitions (all metafeatures)
metafeature_names = {'Metastate Profile' 'Per Subject Likelihood' 'Per Subject Entropy' 'Max Fractional Occupancy' 'FO of metastate 1'}; 
figure;
for i = 2:length(metafeature_store)-1
    subplot(3,2,i-1)
    metafeature = metafeature_store{i};
    imagesc(corr(metafeature,'rows','complete')); colorbar; xlabel('Metafeature of HMM repetition'); ylabel('Metafeature of HMM repetition');
    sgtitle(sprintf('Matrices of correlations between metafeatures of distinct HMM repetitions'))
    title(metafeature_names{i-1})
end

%% Correlation between metafeatures of different repetitions (Principal Components)
figure;
for i = 1:4%size(metafeature_store{1},2)
    subplot(2,2,i)
    metafeature = squeeze(metafeature_store{1}(:,i,:));
    size(metafeature)
    imagesc(corr(metafeature)); colorbar; xlabel('Metafeature of HMM repetition'); ylabel('Metafeature of HMM repetition');
    sgtitle(sprintf('Matrices of correlations between metafeatures of distinct HMM repetitions'))
    title(sprintf('Principal Component %i',i))
end


% %% Multivariate regression
% X = metafeature;
% Y = vars(:,2);
% Y = (Y-min(Y))/(max(Y) - min(Y));
% [beta,Sigma,E,CovB,logL] = mvregress(X,Y);



%% Eigendecomposition
metafeature = squeeze(metafeature_store{1}(:,1,:));
% Real metafeature
[~,D] = eig(corr(metafeature,'rows','complete'));
X = 1:length(D);
Y = diag(D);
figure();
subplot(1,2,1); 
yyaxis left; plot(X,Y,'-x'); ylabel('Size of eigenvalue')
Y_exp = cumsum((Y/sum(Y)) * 100);
yyaxis right; plot(X,Y_exp,'-o');
title('Actual metafeature')
xlabel('Eigenvalue number'); ylabel('Explained variance')
% Simulated metafeature
[V,D] = eig(corr(metafeature_simu(:,:,22),'rows','complete'));
X = 1:length(D);
Y = sort(diag(D),'descend');
subplot(1,2,2);
yyaxis left; plot(X,Y,'-x'); ylabel('Size of eigenvalue')
Y_exp = cumsum((Y/sum(Y)) * 100);
yyaxis right; plot(X,Y_exp,'-o');
title('Simulated metafeature')
xlabel('Eigenvalue number'); ylabel('Explained Variance')
sgtitle(sprintf('Size of eigenvalues of correlation matrix of metafeatures of distinct HMM repetitions'))

% %% Covariances between metafeatures of different repetitions
% figure;
% subplot(1,2,1); imagesc(cov(metafeature,'omitrows')); colorbar; xlabel('Metafeature of HMM repetition'); ylabel('Metafeature of HMM repetition'); title('Real metafeature')
% subplot(1,2,2); imagesc(cov(metafeature_simu(:,:,22),'omitrows')); colorbar; xlabel('Metafeature of HMM repetition'); ylabel('Metafeature of HMM repetition'); title('Simulated metafeature')
% sgtitle(sprintf('Matrices of covarainces between metafeatures (real and simulated) of distinct HMM repetitions'))



%% Plot the total r^2 across all intelligence variables by no. states
figure()
bar(mean(explained_variance,1)')
xlabel('No. states'); ylabel('Mean r^2 (across 34 variables)')
title('Total r^2 across 34 intelligence variables by no. states of HMM runs')
set(gca, 'XTick', 1:length(max(K_all_reps)),'XTickLabel',max(K_all_reps));
 
% %%% Plot the total r^2 across all intelligence variables by no. states (mean across repetitions)
% error_vec_plot = NaN(length(max(K_all_reps))/3,1);
% mean_ev_across_vars = mean(explained_variance);
% for i = 1:length(max(K_all_reps))/3 % 3 repetitions of each number of states
%     error_vec_plot(i) = mean(mean_ev_across_vars(3*i-2:3*i));
% end
% K_vec_plot = (min(max(K_all_reps)):3:max(max(K_all_reps)))'; % 3 repetitions of each number of states
% figure()
% bar(K_vec_plot,error_vec_plot)
% title('Mean total r^2 across HMM runs of the same states for 34 intelligence variables')
% xlabel('Number of states'); ylabel('Mean explained variance (across 34 variables)')







%% Distribution of metafeature (real + simulated)
n_bins = 20;

for rep = 1:n_repetitions
    figure(100)
    sgtitle(sprintf('Distribution of metafeature values'))
    subplot(n_repetitions,1,rep)
    histogram(metafeature_norm(:,rep),n_bins)
    xlabel('Metafeature'); ylabel('Frequency');

    figure(101) % note the simulated metafeature was given a Gaussian distribution and that this just shows for 1 variable the distribution
    sgtitle(sprintf('Distribution of simulated metafeature values'))
    subplot(n_repetitions,1,rep)
    histogram(metafeature_simu(:,rep,1),n_bins)
    xlabel('Metafeature'); ylabel('Frequency');
    
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
n_subjects = size(metafeature,1);
% What's the difference between our simulated metafeature and the real one?
% boxplots for simulated vs actual metafeature
figure();
subplot(2,1,1); boxplot(metafeature_norm); xlabel('HMM repetition');
title('Actual metafeature'); ylabel('Normalized metafeature value');
subplot(2,1,2); boxplot(metafeature_simu(:,:,1)); xlabel('HMM repetition'); % again, just simulated mf for 1 variable
title('Simulated metafeature'); ylabel('Normalized metafeature value');

%%
n_metafeatures = (size(metafeature_norm,2)/n_repetitions);
corr_vars_metafeature_real = NaN(n_var,n_repetitions*n_metafeatures);
corr_vars_metafeature_simu = NaN(n_var,n_repetitions*n_metafeatures);
for v = 1:n_var
   corr_vars_metafeature_real(v,:) =  corr(metafeature,vars(:,v),'rows','complete');
   corr_vars_metafeature_simu(v,:) =  corr(squeeze(metafeature_simu(:,:,v)),vars(:,v),'rows','complete');
end


figure(); 
subplot(1,2,1); imagesc(corr_vars_metafeature_real); colorbar; title('Correlation of metafeature and vars targets'); xlabel('HMM repetition'); ylabel('Variable'); % some variables are really good, others aren't
subplot(1,2,2); imagesc(corr_vars_metafeature_simu); colorbar; title('Correlation of simulated metafeature and vars targets'); xlabel('HMM repetition'); ylabel('Variable');% some variables are really good, others aren't


%% Plot heatmaps of correlations between errors of predictions and metafeatures
all_corr_error_metafeature = NaN(n_var,n_repetitions,4);
all_corr_error_metafeature_simulated = NaN(n_var,n_repetitions,4);
all_corr_error_metafeature_gauss = NaN(n_var,n_repetitions,4);
corr_measures = {'Pearson','Spearman','Kendall','Distance'};

for v = 1:n_var
    [~,~,Metafeature,non_nan_idx] = nan_subject_remove(v,vars_norm,vars_hat_norm,metafeature_norm); % remove NaNs (this needs to be done for every var because different subjects are missing for each var)
    %squared_error = (vars_target - Predictions).^2; % Note squared error
    %metafeature_simu(non_nan_idx,:,v) = metafeature_simulation_creation(squared_error,Metafeature,simulation_options);
    for k = 1:3
        corr_type = corr_measures{k};
        for j = 1:n_repetitions*n_metafeatures;[all_corr_error_metafeature(v,j,k),~] = corr(squared_error(non_nan_idx,j,v),-Metafeature(:,j),'type',corr_type); end
        for j = 1:n_repetitions*n_metafeatures; [all_corr_error_metafeature_simulated(v,j,k),~] = corr(squared_error(non_nan_idx,j,v),-metafeature_simu(non_nan_idx,j,v),'type',corr_type); end
        %for j = 1:n_repetitions; [all_corr_error_metafeature_gauss(v,j,k),~] = corr(squared_error(non_nan_idx,j,v),Metafeature_gaussianized(non_nan_idx,j),'type',corr_type); end

    end
%     % compare distance correlations of real and simulated metafeatures
%     for j = 1:n_repetitions; all_corr_error_metafeature(v,j,4) = distcorr(squared_error(non_nan_idx,j,v),Metafeature(:,j)); end
%     for j = 1:n_repetitions; all_corr_error_metafeature_simulated(v,j,4) = distcorr(squared_error(non_nan_idx,j,v),metafeature_simu(non_nan_idx,j,v)); end
%     %for j = 1:n_repetitions; all_corr_error_metafeature_gauss(v,j,4) = distcorr(squared_error(non_nan_idx,j,v),Metafeature_gaussianized(non_nan_idx,j)); end
end
%%
for j = 1%:3%4
    ax_min = [min(min(all_corr_error_metafeature(:,:,j))),min(min(all_corr_error_metafeature_simulated(:,:,j))),min(min(all_corr_error_metafeature_gauss(:,:,j)))];
    ax_max = [max(max(all_corr_error_metafeature(:,:,j))),max(max(all_corr_error_metafeature_simulated(:,:,j))),max(max(all_corr_error_metafeature_gauss(:,:,j)))];
    figure()
    sgtitle(sprintf('%s Correlation of prediction error vs metafeature',corr_measures{j}))
    subplot(1,2,1); imagesc(all_corr_error_metafeature(:,:,j)); colorbar; caxis([min(ax_min) max(ax_max)]);
    xlabel('HMM repetition (increasing K->)'); ylabel('Variable'); title(sprintf('True metafeature'))
%     subplot(1,3,2); imagesc(all_corr_error_metafeature_gauss(:,:,j)); colorbar; caxis([min(ax_min) max(ax_max)]);
%     xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Gaussianized metafeature'))  
    subplot(1,2,2); imagesc(all_corr_error_metafeature_simulated(:,:,j)); colorbar; caxis([min(ax_min) max(ax_max)]);
    xlabel('HMM repetition (increasing K->)'); ylabel('Variable'); title(sprintf('Simulated metafeature'))

end

%%
% pearson_mf_real  = all_corr_error_metafeature(:,:,1); % pearson
% pearson_mf_sim  = all_corr_error_metafeature_simulated(:,:,1); % pearson
% figure()
% scatter(pearson_mf_real,pearson_mf_sim)
% title('Pearson Correlation of real & simu mf for each of the 34 vars and 5 repetitions')
% xlabel('True metafeature'); ylabel('Simulated metafeature');
% legend('Repetition 1', 'Repetition 2', 'Repetition 3', 'Repetition 4', 'Repetition 5')
% 
% spearman_mf_real = all_corr_error_metafeature(:,:,2); % spearman
% spearman_mf_sim  = all_corr_error_metafeature_simulated(:,:,2); % spearman
% figure()
% scatter(spearman_mf_real,spearman_mf_sim)
% title('Spearman Correlation of real & simu mf for each of the 34 vars and 5 repetitions')
% xlabel('True metafeature'); ylabel('Simulated metafeature');
% legend('Repetition 1', 'Repetition 2', 'Repetition 3', 'Repetition 4', 'Repetition 5')

%% Display tables of variables
load('intelligence_variables.mat')
T = cell2table(intelligence_variables);
T.Properties.VariableNames = {'Variable Number' 'Variable ID' 'Variable Category'};
figure;
uitable('Data',T{:,:},'ColumnName',T.Properties.VariableNames,'RowName',T.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1],'ColumnWidth',{100, 100, 200});


%% Plot simulations of increasing rho
load('accuracy_by_rho.mat')
X = rho_vec;
figure();
Y_smooth = reshape(smooth(ev_rho_store),[200 6]);
Y_smooth_clean = [ev_rho_store(1:2,:); Y_smooth(3:end-2,:);  ev_rho_store(end-1:end,:)];
Y = Y_smooth_clean;
% Y = ev_rho_store;
plot(X,Y); hold on % simulated metafeature
scatter(0,sum(EV_all(:,2,1)),'r') % no metafeature (we take no_metafeature_ev(1) but could take any element - see issues at top of doc regarding 'no metafeatures' redundancy)
% scatter(true_corr,sum(EV_all(:,4,2)),'x','r') % real metafeature
% scatter(true_corr,sum(EV_all(:,4,1)),'x','r') % simulated metafeature
scatter(mean(mean(abs(all_corr_error_metafeature(:,:,1)))),sum(EV_all(:,4,2)),'x','r') % real metafeature
scatter(mean(mean(abs(all_corr_error_metafeature_simulated(:,:,1)))),sum(EV_all(:,4,1)),'x','k') % simulated metafeature

mean(mean(abs(all_corr_error_metafeature(:,:,1))))

title('Accuracy of stacked predictions when simulating {\itn} metafeatures');
xlabel('Correlation of simulated metafeature and accuracy of HMM run predictions, \rho');
ylabel('Total explained variance (across 34 variables)');
legend('FWLS Ridge (n = 1)','FWLS Ridge (n = 2)','FWLS Ridge (n = 3)','FWLS Ridge (n = 4)','FWLS Ridge (n = 5)','FWLS Ridge (n = 6)','Stacked Ridge (no metafeature)','FWLS Ridge (true metafeature)','FWLS Ridge (simulated metafeature)');





%% Statistical summaries of metafeature (real + simulated)
% for v = 1
% mean_mf = [mean(metafeature_norm); mean(metafeature_simu(non_nan_idx,:,v))]
% std_mf = [std(metafeature_norm); std(metafeature_simu(non_nan_idx,:,v))]
% kurt_mf = [kurtosis(metafeature_norm); kurtosis(metafeature_simu(non_nan_idx,:,v))]
% skew_mf = [skewness(metafeature_norm); skewness(metafeature_simu(non_nan_idx,:,v))]
% end




%%
% To do: mess about with metafeatures simulation to see what works and what
% doesn't, e.g.
% - can we introduce a nonlinear relationship and still see improvements? Might need a nonlinear method


% n_outliers_vars = [(1:34)' NaN(34,1)];
% n_unique_vars = [(1:34)' NaN(34,1)];
% n_nan_vars = [(1:34)' NaN(34,1)];
% for i = 1:34; n_unique_vars(i,2) = sum(~isnan((unique(vars(:,i))))); end % check to see if discrete vs continuous (vs continuous but repeated values common)
% for i = 1:34; n_nan_vars(i,2) = sum(isnan(vars(:,i))); end % check NaNs in data
% for i = 1:34; n_outliers_vars(i,2) = sum(isoutlier(vars(:,i))); end % check outliers in data






   

% function [Entropy_gaussianized,Likelihood_gaussianized] = gaussianize_r()
% Here I am trying to call an R script to Gaussianize the metafeatures
%     setenv('PATH', getenv('PATH')+":/usr/local/bin")
% see % https://stackoverflow.com/questions/38456144/rscript-command-not-found
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


% 
% no_mf_mse = sum(MSE_stack_all(:,2,2));
% no_mf_ev = sum(EV_all(:,2,2));
% true_corr = mean(mean(abs(pearson_error_metafeature)));
% 
% true_mf_mse = sum(MSE_stack_all(:,4,2));
% true_mf_ev = sum(EV_all(:,4,2));
% 
% 
% % I want stacked ridge without metafaetue, then real metafeature, then
% % simualted
% 
% X = rho_vec;
% Y = ev_rho_store;
% figure(); plot(X,Y); hold on % simulated metafeature
% scatter(true_corr,true_mf_ev,'x','k') % real metafeature
% scatter(0,no_mf_ev,'r') % no metafeature
% title('Accuracy of stacked predictions when simulating metafeatures');
% xlabel('Correlation of simulated metafeature and accuracy of HMM run predictions, \rho');
% ylabel('Total explained variance (across 34 variables)');
% legend('FWLS Ridge (simulated metafeature)','FWLS Ridge (true metafeature)','Stacked Ridge (without metafeatures)');
% Y = mpe_rho_store;
% figure(); plot(X,Y); hold on % simulated metafeature
% scatter(true_corr,true_mf_mse,'x','k') % real metafeature
% scatter(0,no_mf_mse,'r')  % no metafeature
% title('Accuracy of stacked predictions when simulating metafeatures');
% xlabel('Correlation of simulated metafeature and accuracy of HMM run predictions, \rho');
% ylabel('Total mean squared error');
% legend('FWLS Ridge (simulated metafeature)','FWLS Ridge (true metafeature)','Stacked Ridge (without metafeatures)');


%load([ DirOut 'HMMs_meta_data_GROUP_gaussianized.mat']); % load metadata
%load([ DirOut 'HMMs_meta_data_subject_gaussianized']) % load Gaussianized metadata
%[Entropy_gaussianized,Likelihood_gaussianized] = gaussianize_r(); % Gaussianize metafeatures (using script in R) (note: must change the save directory to the correct folder in R)

