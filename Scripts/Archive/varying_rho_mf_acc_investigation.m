%% SIMULATED DATA
clc; clear;
rng('default') % set for reproducibility
DirOut = ['/Users/bengriffin/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps/']; % Test folder
load([ DirOut  'HMM_predictions.mat']) % load predictions
load([ DirOut  'vars_target.mat']) % load targets

n_outliers_vars = [(1:34)' NaN(34,1)]; %n_unique_vars = [(1:34)' NaN(34,1)]; n_nan_vars = [(1:34)' NaN(34,1)];
%for i = 1:34; n_unique_vars(i,2) = sum(~isnan((unique(vars(:,i))))); end % check to see if discrete vs continuous (vs continuous but repeated values common)
%for i = 1:34; n_nan_vars(i,2) = sum(isnan(vars(:,i))); end % check NaNs in data
for i = 1:34; n_outliers_vars(i,2) = sum(isoutlier(vars(:,i))); end

%%%%%%%% CAN I DO THIS BECAUSE THEY ARE SHIFTING THEM BY DIFFERENT AMOUNTS
%%%%%%%% SO NOT SURE THIS IS LEGIT? MAYBE TRY SHIFTING THEM BY THE SAME
%%%%%%%% AMOUNT
%%% Min-max scaling is sensitive to outliers (if we have a MASSIVE max or V
%%% SMALL min, then this will bunch up the rest
% SHOULD WE THEREFORE REMOVE OUTLIERS BEFORE NORMALIZING
vars = (vars-min(vars))./(max(vars)-min(vars));
vars_hat = (vars_hat-min(vars_hat))./(max(vars_hat)-min(vars_hat));
% WHY DOES NORMALIZING NOT WORK?
%vars = normalize(vars);
%vars_hat = normalize(vars_hat);


% load and store metadata
load('HMMs_meta_data_subject_r_1_27_40_63')
load('HMMs_meta_data_subject_r_1_27_40_63_gaussianized')
load('Metafeature_correlations/metafeatures_correlations.mat')
load('Metafeature_correlations/metafeatures_correlations_norm.mat')
load('Metafeature_correlations/metafeatures_gaussianized_correlations_norm.mat')
%Entropy_subject_all = Entropy_subject_all-min(Entropy_subject_all))./(max(Entropy_subject_all)-min(Entropy_subject_all)); (Update: this does nothing)
%Entropy_gaussianized = (Entropy_gaussianized-min(Entropy_gaussianized))./(max(Entropy_gaussianized)-min(Entropy_gaussianized));
Likelihood_subject_all = (likelihood_subject_all-min(likelihood_subject_all))./(max(likelihood_subject_all)-min(likelihood_subject_all));
Likelihood_gaussianized = (Likelihood_gaussianized-min(Likelihood_gaussianized))./(max(Likelihood_gaussianized)-min(Likelihood_gaussianized));


% Intialise variables
n_subjects = size(vars_hat,1); n_var = size(vars_hat,2); n_repetitions = 5; 
explained_variance_ST = NaN(n_var,n_repetitions); MSE_ST = NaN(n_var,n_repetitions);
explained_variance_stack_ls = NaN(n_var,1); explained_variance_stack_ridge = NaN(n_var,1);
explained_variance_FWLS_ls = NaN(n_var,1); explained_variance_FWLS_ridge = NaN(n_var,1);
MSE_stack_ls = NaN(n_var,1); MSE_stack_ridge = NaN(n_var,1);
MSE_FWLS_ls = NaN(n_var,1); MSE_FWLS_ridge = NaN(n_var,1);
prediction_stack_ls = NaN(n_subjects,n_var); prediction_stack_ridge = NaN(n_subjects,n_var);
prediction_FWLS_ls = NaN(n_subjects,n_var); prediction_FWLS_ridge = NaN(n_subjects,n_var);

% intialise variables for investigation of effect of increasing rho
p_entropy_vec = 0.01:0.01:1;
total_MSE_ST = NaN(length(p_entropy_vec),n_repetitions);
total_MSE_stack_ridge = NaN(length(p_entropy_vec),1);
total_MSE_FWLS_ridge = NaN(length(p_entropy_vec),1);
total_EV_ST = NaN(length(p_entropy_vec),n_repetitions);
total_EV_stack_ridge = NaN(length(p_entropy_vec),1);
total_EV_FWLS_ridge = NaN(length(p_entropy_vec),1);
pearson_error_entropy_simulated = NaN(34,5);
pearson_error_entropy_actual = NaN(34,5);

for i = 1:length(p_entropy_vec) % set desired correlation
    p_entropy_store = p_entropy_vec(i)
    for var = 1:34
        var;
        
        % Store data for variable
        Predictions = squeeze(vars_hat(:,var,[2 6 14 25 44]));
        vars_target = vars(:,var);
        Entropy = Entropy_subject_all(:,[2 6 14 25 44]); Entropy_gauss = Entropy_gaussianized(:,[2 6 14 25 44]);
        Likelihood = likelihood_subject_all(:,[2 6 14 25 44]); Likelihood_gauss = Likelihood_gaussianized(:,[2 6 14 25 44]);
        squared_error_gaussianized = Squared_error_gaussianized(:,:,var);
        
        % Remove subjects with missing values
        non_nan_idx = find(~isnan(vars_target));
        which_nan = isnan(vars_target);
        if any(which_nan)
            vars_target = vars_target(~which_nan);
            Predictions = Predictions(~which_nan,:);
            Entropy = Entropy(~which_nan,:); Entropy_gauss = Entropy_gauss(~which_nan,:); 
            Likelihood = Likelihood(~which_nan,:); Likelihood_gauss = Likelihood_gauss(~which_nan,:);
            squared_error_gaussianized = squared_error_gaussianized(~which_nan,:);
            warning('NaN found on Yin, will remove...')
        end

%         % Remove subjects with that are outliers
%         non_outlier_idx = find(~isoutlier(vars_target));
%         which_outlier = isoutlier(vars_target);
%         if any(which_outlier)
%             vars_target = vars_target(~which_outlier);
%             Predictions = Predictions(~which_outlier,:);
%             Entropy = Entropy(~which_outlier,:); Entropy_gauss = Entropy_gauss(~which_outlier,:); 
%             Likelihood = Likelihood(~which_outlier,:); Likelihood_gauss = Likelihood_gauss(~which_outlier,:);
%             squared_error_gaussianized = squared_error_gaussianized(~which_outlier,:);
%             warning('NaN found on Yin, will remove...')
%         end    
        
        %vars = (vars-min(vars))./(max(vars)-min(vars));
        %vars_hat = (vars_hat-min(vars_hat))./(max(vars_hat)-min(vars_hat));

        
        
        % determine accuracy of predictions (we want our metafeature to be correlated with this)
        squared_error = (vars_target - Predictions).^2;
        
        %%%%%%%%%% METAFEATURE ARRAY SIMULATION %%%%%%%%%%%%%%
        %p_entropy = repmat(-0.05,1,5); % set desired correlation
        %p_entropy = pearson_error_entropy(var,:) + 0.6; % HERE WE SIMULATE ENTROPY WITH THE SAME CORRELATION AS THE TRUE METAFEATURE
        %p_entropy = pearson_error_entropy_norm(var,:);
        %p_entropy = pearson_error_entropy_gauss_norm(var,:); % let's simulate entropy with same correlation as true mf gaussianized
        p_entropy = repmat(p_entropy_store,1,5);
        
        entropy_simu = metafeature_simulation_creation(squared_error,p_entropy,Entropy); % simulate entropy
        entropy_simu_norm = entropy_simu; % 
        %entropy_simu_norm = (entropy_simu - min(entropy_simu))./(max(entropy_simu) - min(entropy_simu));
        %entropy_simu_norm = (entropy_simu - mean(entropy_simu))+1; % standardize metafeatures to set mean to 1   %./std(entropy_simu)+1;
        metafeature_array = [repmat(ones(size(Entropy)),1,1) entropy_simu_norm];
        %metafeature_array = [repmat(ones(size(Entropy)),1,1) Entropy];
        %%%%%%%%%% METAFEATURE ARRAY SIMULATION END %%%%%%%%%%%%%%
        for j = 1:5; pearson_error_entropy_simulated(var,j) = corr(squared_error(:,j),entropy_simu(:,j)); end
        for j = 1:5; pearson_error_entropy_actual(var,j) = corr(squared_error(:,j),Entropy(:,j)); end
        
        % Let's try the actual metafeature
        %Entropy = (Entropy-min(Entropy))./(max(Entropy)-min(Entropy));
        %metafeature_array = [repmat(ones(size(Entropy)),1,1) Entropy];
        
        
        % Make predictions
        [pred_ST,pred_stack_ls,pred_stack_rdg,pred_FWLS_ls,pred_FWLS_rdg] = predictPhenotype_mf_simulation_V2(Predictions,vars_target,metafeature_array);

        % Note prediction accuracies
        [mse_ST,mse_stack_ls,mse_stack_rdg,mse_FWLS_ls,mse_FWLS_rdg,ev_ST,ev_stack_ls,ev_stack_rdg,ev_FWLS_ls,ev_FWLS_rdg] = ...
            prediction_accuracy_stats(vars_target,pred_ST,pred_stack_ls,pred_stack_rdg,pred_FWLS_ls,pred_FWLS_rdg);
        
        % store prediction accuracies
        MSE_ST(var,:) = mse_ST;
        MSE_stack_ls(var) = mse_stack_ls;
        MSE_stack_ridge(var) = mse_stack_rdg;
        MSE_FWLS_ls(var) = mse_FWLS_ls;
        MSE_FWLS_ridge(var) = mse_FWLS_rdg;
        
        explained_variance_ST(var,:) = ev_ST;
        explained_variance_stack_ls(var) = ev_stack_ls;
        explained_variance_stack_ridge(var) = ev_stack_rdg;
        explained_variance_FWLS_ls(var) = ev_FWLS_ls;
        explained_variance_FWLS_ridge(var) = ev_FWLS_rdg;
        
        
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         % PLOT SIMULATED RELATIONSHIPS BETWEEN PREDICTIVE ACCURACY AND METAFEATURES
%         figure(1) 
%         for rep = 1%:5
%             mf = entropy_simu(:,rep);
%             [pearson_corr,pval_pearson] = corr(mf,squared_error(:,rep));
%             [spearman_corr,pval_spearman] = corr(mf,squared_error(:,rep),'Type','Spearman');
%             subplot(6,6,var)
%             scatter(mf,squared_error(:,rep)); hold on
%         end
%         xlabel('Simulated metafeature (entropy)') ;ylabel('Prediction error')
%         sgtitle(sprintf('SIMULATION - metafeature value vs squared error per subject (\\rho = %.2f)',p_entropy))
%         title(sprintf('Var #%d, P %.2f, S %.2f', var, pearson_corr, spearman_corr));
%         if pval_pearson<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f', var, pearson_corr, spearman_corr)); end; if pval_spearman<0.05; title(sprintf('Var #%d, P %.2f, S %.2f*', var, pearson_corr, spearman_corr)); end; if pval_pearson<0.05 && pval_spearman<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f*', var, pearson_corr, spearman_corr)); end;
%   
%         % PLOT ACTUAL RELATIONSHIPS BETWEEN PREDICTIVE ACCURACY AND METAFEATURES
%         figure(2)
%         for rep = 1%:5
%             mf = Likelihood_gauss(:,rep); %mf = Entropy(:,rep);
%             sq_er = squared_error(:,rep); % sq_er = squared_error_gaussianized(:,rep); 
%             [pearson_corr,pval_pearson] = corr(mf,sq_er);
%             [spearman_corr,pval_spearman] = corr(mf,sq_er,'Type','Spearman');
%             subplot(6,6,var)
%             scatter(mf,sq_er);
%             hold on
%         end
%         
%         xlabel('Likelihood'); ylabel('Prediction error')
%         sgtitle(sprintf('Metafeature value vs squared error per subject (for a single repetition of HMM), P = Pearson, S = Spearman, * = Significant'))
%         title(sprintf('Var #%d, P %.2f, S %.2f', var, pearson_corr, spearman_corr));
%         if pval_pearson<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f', var, pearson_corr, spearman_corr)); end
%         if pval_spearman<0.05; title(sprintf('Var #%d, P %.2f, S %.2f*', var, pearson_corr, spearman_corr)); end
%         if pval_pearson<0.05 && pval_spearman<0.05; title(sprintf('Var #%d, P %.2f*, S %.2f*', var, pearson_corr, spearman_corr)); end
%           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    
    
    total_MSE_ST(i,:) = sum(MSE_ST);
    total_MSE_stack_ridge(i) = sum(MSE_stack_ridge);
    total_MSE_FWLS_ridge(i) = sum(MSE_FWLS_ridge);
    
    total_EV_ST(i,:) = sum(explained_variance_ST);
    total_EV_stack_ridge(i) = sum(explained_variance_stack_ridge);
    total_EV_FWLS_ridge(i) = sum(explained_variance_FWLS_ridge);
end
%save('simulated_prediction_accuracies_enrtopy.mat','total_prediction_accuracy_ST','total_prediction_accuracy_stack','total_prediction_accuracy_FWLS','p_entropy_vec')
%%
figure()
plot(p_entropy_vec,total_MSE_FWLS_ridge)
hold on
scatter(0,sum(MSE_stack_ridge),'x','r')
legend('FWLS Ridge','Stacked Ridge (without metafeature)')
title('Accuracy of stacked predictions when simulating metafeatures')
xlabel('Correlation of simulated metafeature and accuracy of HMM run predictions, \rho'); ylabel('Total Mean Squared Error (across 34 variables)')

figure()
plot(p_entropy_vec,total_EV_FWLS_ridge)
hold on
scatter(0,sum(explained_variance_stack_ridge),'x','r')
legend('FWLS Ridge','Stacked Ridge (without metafeature)')
title('Accuracy of stacked predictions when simulating metafeatures')
xlabel('Correlation of simulated metafeature and accuracy of HMM run predictions, \rho'); ylabel('Total Explained Variance (across 34 variables)')



%%
%%%%%%%%% PLOT SIMULATION RESULTS %%%%%%%%%%%%%%
%%% EXPLAINED VARIANCE
X = 1:34; % no. variables
figure()
%scatter(1:n_var,explained_variance_stack_ls,'o','r')
hold on
scatter(1:n_var,explained_variance_stack_ridge,'o','g'); hold on
%scatter(1:n_var,explained_variance_FWLS_ls,'o','m')
scatter(1:n_var,explained_variance_FWLS_ridge,'o','k')
%scatter(X,predictions_accuracy,'x','b')
for rep = 1:n_repetitions
    scatter(1:n_var,explained_variance_ST(:,rep),'x','b')
end
legend('Stacked Ridge','FWLS Ridge','HMM reps')
%legend('Stacked LS','Stacked Ridge','FWLS LS','FWLS Ridge','HMM reps')
title(sprintf('Explained variance, simulated \\rho = %1.2f', p_entropy(1)))
xlabel('Variable'); ylabel('Explained Variance');


%%% MEAN PREDICTION ERROR
X = 1:34; % no. variables
figure()
% scatter(1:n_var,prediction_accuracy_stack_ls,'o','r')
hold on
scatter(1:n_var,MSE_stack_ridge,'o','g'); hold on
%scatter(1:n_var,prediction_accuracy_FWLS_ls,'o','m')
scatter(1:n_var,MSE_FWLS_ridge,'o','k')
%scatter(X,predictions_accuracy,'x','b')
for rep = 1:5
    scatter(1:n_var,MSE_ST,'x','b')
    %scatter(1:n_var,squeeze(Prediction_accuracy_2(:,:,rep)),'x','b')
end
legend('Stacked Ridge','FWLS Ridge','HMM reps')
%legend('Stacked LS','Stacked Ridge','FWLS LS','FWLS Ridge','HMM reps')
title(sprintf('Mean prediction error, \\rho = %1.2f', p_entropy(1)))
xlabel('Variable'); ylabel('Mean Prediction Error');



function meta_simu = metafeature_simulation_creation(pred_acc,p_meta,Metafeature)
    n_subjects = size(pred_acc,1);
    v_meta = randn(n_subjects, 1); % create random variable 'v'
    u_meta = (pred_acc ./ std(pred_acc)) - mean(pred_acc); % reverse: x = (s1 * u + m1)' to get u = randn(1, n);
    y_meta = std(Metafeature) .* squeeze((p_meta .* u_meta + sqrt(1 - p_meta.^2) .* v_meta));
    y_meta = y_meta + mean(Metafeature) - mean(y_meta); % set mean of rv to desired mean (Update: this doesn't make a difference)
    meta_simu = y_meta;
end
