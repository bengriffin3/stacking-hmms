%% SIMULATING METAFEATURES
clear; clc; close all;
rng('default') % set seed for reproducibility
DirOut = '/Users/au699373/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_vary_states_3_14/'; % Set directory

% load data
load([ DirOut  'HMM_predictions.mat']) % load predictions
load('vars_target.mat') % load targets
load([ DirOut 'HMMs_meta_data_subject_NEW']);
load([ DirOut 'HMMs_meta_data_GROUP_NEW']); % load metadata

%%

%load([ DirOut 'HMMs_meta_data_subject_gaussianized']) % load Gaussianized metadata
%[Entropy_gaussianized,Likelihood_gaussianized] = gaussianize_r(); % Gaussianize metafeatures (using script in R) (note: must change the save directory to the correct folder in R)

% Note dimensions of arrays
n_var = size(vars_hat,2);
n_repetitions = size(vars_hat,3);

% Select simulation choices
simulation_options.rep_selection = 1:n_repetitions;%[2 6 14 25 44];  Select repetitions to use for simulation (default is all repetitions)
simulation_options.n_folds = 10;
simulation_options.corr_specify = []; % specify correlation between simulated metafeature and accuracy of HMM prediction (if empty, we use same correlation as true metafeatures)


% select metafeature e.g. Entropy/Likelihood
metafeature_store = {FO_metastate1};%{likelihood_subject Entropy_subject maxFO_all_reps switchingRate_all_reps};
metafeature = metafeature_store{1};
%Metafeature_all = likelihood_subject; %Metafeature_all = Entropy_subject;
%Metafeature_gaussianized = Likelihood_gaussianized;

% normalize variables and target (should we remove outliers before normalizing since this will change max/min?
%  if we normalize each prediction, then we stack the predictions,
% how do we de-normalize? since we change the predictions by different
% amounts
vars_norm =  (vars-min(vars))./(max(vars)-min(vars));
vars_hat_norm = (vars_hat-min(vars_hat))./(max(vars_hat)-min(vars_hat));
metafeature_norm = (metafeature-min(metafeature))./(max(metafeature)-min(metafeature)); % (Update: this does nothing - the weights just basically become very very small. If the metafeature values were super small, then this might matter because they is an upper limit on the weights)

% Run simulation
[pred_ST,pred_stack_all,MSE_stack_ST,MSE_stack_all,EV_ST,EV_all,pearson_error_metafeature,metafeature_simu] = simulation_full(vars_norm,vars_hat_norm,metafeature_norm,simulation_options);



% Plot simulation results
    
%%% EXPLAINED VARIANCE
% 2-dim: (1) LSQ (2) Ridge (3) FWLS LSQ (4) FWLS Ridge
% 3-dim: (1) Simulated (2) True metafeature
figure(1)
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

%%% MEAN PREDICTION ERROR
% 2-dim: (1) LSQ (2) Ridge (3) FWLS LSQ (4) FWLS Ridge
% 3-dim: (1) Simulated (2) True metafeature
figure(2)
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







%% Distribution of metafeature (real + simulated)
n_bins = 20;

for rep = 1:5
    figure(3)
    sgtitle(sprintf('Distribution of metafeature values'))
    subplot(5,1,rep)
    histogram(metafeature(:,rep),n_bins)
    xlabel('Metafeature'); ylabel('Frequency');

    figure(4) % note the simulated metafeature was given a Gaussian distribution
    sgtitle(sprintf('Distribution of simulated metafeature values'))
    subplot(5,1,rep)
    histogram(metafeature_simu(:,rep),n_bins)
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
subplot(2,1,1); boxplot(metafeature); xlabel('HMM repetition');
title('Actual metafeature'); ylabel('Normalized metafeature value');
subplot(2,1,2); boxplot(metafeature_simu(:,:,1)); xlabel('HMM repetition');
title('Simulated metafeature'); ylabel('Normalized metafeature value');

%% Plot heatmaps of correlations between errors of predictions and metafeatures
all_corr_error_metafeature = NaN(n_var,n_repetitions,4);
all_corr_error_metafeature_simulated = NaN(n_var,n_repetitions,4);
all_corr_error_metafeature_gauss = NaN(n_var,n_repetitions,4);
corr_measures = {'Pearson','Spearman','Kendall','Distance'};

for v = 1:n_var
    [vars_target,Predictions,Metafeature,non_nan_idx] = nan_subject_remove(v,vars_norm,vars_hat_norm,metafeature_norm); % remove NaNs
    squared_error = (vars_target - Predictions).^2; % Note squared error
    metafeature_simu(non_nan_idx,:,v) = metafeature_simulation_creation(squared_error,Metafeature,simulation_options);
    for k = 1:3
        corr_type = corr_measures{k};
        for j = 1:n_repetitions;[all_corr_error_metafeature(v,j,k),~] = corr(squared_error(:,j),Metafeature(:,j),'type',corr_type); end
        for j = 1:n_repetitions; [all_corr_error_metafeature_simulated(v,j,k),~] = corr(squared_error(:,j),metafeature_simu(non_nan_idx,j,v),'type',corr_type); end
        %for j = 1:n_repetitions; [all_corr_error_metafeature_gauss(v,j,k),~] = corr(squared_error(:,j),Metafeature_gaussianized(non_nan_idx,j),'type',corr_type); end
    end
    % compare distance correlations of real and simulated metafeatures
    for j = 1:n_repetitions; all_corr_error_metafeature(v,j,4) = distcorr(squared_error(:,j),Metafeature(:,j)); end
    for j = 1:n_repetitions; all_corr_error_metafeature_simulated(v,j,4) = distcorr(squared_error(:,j),metafeature_simu(non_nan_idx,j,v)); end
    %for j = 1:n_repetitions; all_corr_error_metafeature_gauss(v,j,4) = distcorr(squared_error(:,j),Metafeature_gaussianized(non_nan_idx,j)); end
end

for j = 1:4
    ax_min = [min(min(all_corr_error_metafeature(:,:,j))),min(min(all_corr_error_metafeature_simulated(:,:,j))),min(min(all_corr_error_metafeature_gauss(:,:,j)))];
    ax_max = [max(max(all_corr_error_metafeature(:,:,j))),max(max(all_corr_error_metafeature_simulated(:,:,j))),max(max(all_corr_error_metafeature_gauss(:,:,j)))];
    figure()
    sgtitle(sprintf('%s Correlation of prediction error vs metafeature',corr_measures{j}))
    subplot(1,2,1); imagesc(all_corr_error_metafeature(:,:,j)); colorbar; caxis([min(ax_min) max(ax_max)]);
    xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('True metafeature'))
%     subplot(1,3,2); imagesc(all_corr_error_metafeature_gauss(:,:,j)); colorbar; caxis([min(ax_min) max(ax_max)]);
%     xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Gaussianized metafeature'))  
    subplot(1,2,2); imagesc(all_corr_error_metafeature_simulated(:,:,j)); colorbar; caxis([min(ax_min) max(ax_max)]);
    xlabel('HMM repetition'); ylabel('Variable'); title(sprintf('Simulated metafeature'))

end

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


