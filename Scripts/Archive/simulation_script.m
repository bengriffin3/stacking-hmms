%% SIMULATED DATA
clc; clear;
rng('default') % set for reproducibility

% load predictions, targets, and metadata
load('HMM_predictions.mat')
load('vars_target.mat')
vars = (vars-min(vars))./(max(vars)-min(vars));
vars_hat = (vars_hat-min(vars_hat))./(max(vars_hat)-min(vars_hat));
% load and store metadata
load('HMMs_meta_data_subject_r_1_27_40_63')
load('HMMs_meta_data_subject_r_1_27_40_63_gaussianized')
Entropy_subject_all = (Entropy_subject_all-min(Entropy_subject_all))./(max(Entropy_subject_all)-min(Entropy_subject_all));
Entropy_gaussianized = (Entropy_gaussianized-min(Entropy_gaussianized))./(max(Entropy_gaussianized)-min(Entropy_gaussianized));
Likelihood_subject_all = (likelihood_subject_all-min(likelihood_subject_all))./(max(likelihood_subject_all)-min(likelihood_subject_all));
Likelihood_gaussianized = (Likelihood_gaussianized-min(Likelihood_gaussianized))./(max(Likelihood_gaussianized)-min(Likelihood_gaussianized));

% Intialise variables
n_subjects = size(vars_hat,1); n_var = size(vars_hat,2); n_repetitions = 5; 
explained_variance_ST = NaN(n_var,n_repetitions);
explained_variance_stack_ls = NaN(n_var,1);
explained_variance_stack_ridge = NaN(n_var,1);
explained_variance_FWLS_ls = NaN(n_var,1);
explained_variance_FWLS_ridge = NaN(n_var,1);
MSE_ST = NaN(n_var,n_repetitions);
MSE_stack_ls = NaN(n_var,1);
MSE_stack_ridge = NaN(n_var,1);
MSE_FWLS_ls = NaN(n_var,1);
MSE_FWLS_ridge = NaN(n_var,1);
prediction_stack_ls = NaN(n_subjects,n_var);
prediction_stack_ridge = NaN(n_subjects,n_var);
prediction_FWLS_ls = NaN(n_subjects,n_var);
prediction_FWLS_ridge = NaN(n_subjects,n_var);

% explained variance seems to go up but prediction error isn't going down -
% what's going on?
p_entropy_vec = 0.8;%0:0.01:1;
total_MSE_ST = NaN(length(p_entropy_vec),n_repetitions);
total_MSE_stack_ridge = NaN(length(p_entropy_vec),1);
total_MSE_FWLS_ridge = NaN(length(p_entropy_vec),1);
total_EV_ST = NaN(length(p_entropy_vec),n_repetitions);
total_EV_stack_ridge = NaN(length(p_entropy_vec),1);
total_EV_FWLS_ridge = NaN(length(p_entropy_vec),1);

for i = 1:length(p_entropy_vec) % set desired correlation
    p_entropy = p_entropy_vec(i)
    for var = 1:34
        var;
        
        % Store data for variable
        Predictions = squeeze(vars_hat(:,var,[2 6 14 25 44]));
        vars_target = vars(:,var);
        Entropy = Entropy_subject_all(:,[2 6 14 25 44]); Entropy_gauss = Entropy_gaussianized(:,[2 6 14 25 44]);
        Likelihood = likelihood_subject_all(:,[2 6 14 25 44]); Likelihood_gauss = Likelihood_gaussianized(:,[2 6 14 25 44]);
        
        % Remove subjects with missing values
        non_nan_idx = find(~isnan(vars_target));
        which_nan = isnan(vars_target);
        if any(which_nan)
            vars_target = vars_target(~which_nan);
            Predictions = Predictions(~which_nan,:);
            Entropy = Entropy(~which_nan,:); Entropy_gauss = Entropy_gauss(~which_nan,:); 
            Likelihood = Likelihood(~which_nan,:); Likelihood_gauss = Likelihood_gauss(~which_nan,:); 
            warning('NaN found on Yin, will remove...')
        end
        
        % determine accuracy of predictions (we want our metafeature to be correlated with this)
        squared_error = (vars_target - Predictions).^2;
        
        %%%%%%%%%% METAFEATURE ARRAY SIMULATION %%%%%%%%%%%%%%
        %p_entropy = 0.5; % set desired correlation
        entropy_simu = metafeature_simulation_creation(squared_error,p_entropy,Entropy); % simulate entropy
        entropy_simu_norm = entropy_simu; % entropy_simu_norm = (entropy_simu - mean(entropy_simu))+1; % standardize metafeatures to set mean to 1   %./std(entropy_simu)+1; %entropy_simu_norm = (entropy_simu - min(entropy_simu))./(max(entropy_simu) - min(entropy_simu));
        metafeature_array = [repmat(ones(size(Entropy)),1,1) entropy_simu_norm];
        %%%%%%%%%% METAFEATURE ARRAY SIMULATION END %%%%%%%%%%%%%%
        
        % Let's try the actual metafeature
        %metafeature_array = [repmat(ones(size(Entropy)),1,1) Entropy];
        
        % Make predictions (and note mean squared errors/explained variances)
        [mse_ST,mse_stack_ls,mse_stack_rdg,mse_FWLS_ls,mse_FWLS_rdg,ev_ST,ev_stack_ls,ev_stack_rdg,ev_FWLS_ls,ev_FWLS_rdg,~,~,~,~] = ...
            predictPhenotype_mf_simulation_V2(Predictions,vars_target,metafeature_array);
        % corrcoef(Prediction_accuracy(:,1),entropy_simu(:,1)) % check correlation is as we set
        
        %     % store predictions and accuracy measures
        %     prediction_stack(non_nan_idx,var) = p_stack;
        %     prediction_FWLS(non_nan_idx,var) = p_FWLS;
        
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
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % PLOT SIMULATED RELATIONSHIPS BETWEEN PREDICTIVE ACCURACY AND METAFEATURES
%         figure(1) 
%         for rep = 1%:5
%             Entropy_plot = entropy_simu(:,rep);
%             squared_error_plot = squared_error(:,rep); % corrcoef(Prediction_accuracy_plot,Entropy_plot) % check correlation is as we set
%             subplot(6,6,var)
%             scatter(Entropy_plot,squared_error_plot); hold on
%         end
%         xlabel('Simulated metafeature (entropy)') ;ylabel('Prediction error')
%         sgtitle(sprintf('SIMULATION - metafeature value vs squared error per subject (\\rho = %.2f)',p_entropy))
%         title(sprintf('Variable #%d', var));
        
        % PLOT ACTUAL RELATIONSHIPS BETWEEN PREDICTIVE ACCURACY AND METAFEATURES
        figure(2)
        for rep = 1%:5
            mf = Likelihood(:,rep); %mf = Likelihood(:,rep);
            pearson_corr = corr(mf,squared_error(:,rep)); %pearson_corr = corr(Entropy_gauss(:,rep),squared_error(:,rep));
            spearman_corr = corr(mf,squared_error(:,rep),'Type','Spearman');
            subplot(6,6,var)
            scatter(mf,squared_error(:,rep));
            %scatter(Likelihood_plot,squared_error_plot);
            hold on
        end
        xlabel('Likelihood'); ylabel('Prediction error')
        sgtitle(sprintf('Metafeature value vs squared error per subject (for a single repetition of HMM), P = Pearson, S = Spearman'))
        title(sprintf('Var #%d, P %.2f, S %.2f', var, pearson_corr, spearman_corr));%, pearson_error_entropy(var,1), spearman_error_entropy(var,1)));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    %corrcoef(Prediction_accuracy(:,1),entropy_simu(:,1));
    total_MSE_ST(i,:) = sum(MSE_ST);
    total_MSE_stack_ridge(i) = sum(MSE_stack_ridge);
    total_MSE_FWLS_ridge(i) = sum(MSE_FWLS_ridge);
    
    total_EV_ST(i,:) = sum(explained_variance_ST);
    total_EV_stack_ridge(i) = sum(explained_variance_stack_ridge);
    total_EV_FWLS_ridge(i) = sum(explained_variance_FWLS_ridge);
end
%save('simulated_prediction_accuracies_enrtopy.mat','total_prediction_accuracy_ST','total_prediction_accuracy_stack','total_prediction_accuracy_FWLS','p_entropy_vec')


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
title(sprintf('Explained variance, simulated \\rho = %1.2f', p_entropy))
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
title(sprintf('Mean prediction error, \\rho = %1.2f', p_entropy))
xlabel('Variable'); ylabel('Mean Prediction Error');

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





%%% likelihood metafeature code to add
%    entropy_simu_norm = (entropy_simu - mean(entropy_simu))+1;%./std(entropy_simu)+1; %entropy_simu_norm = (entropy_simu - min(entropy_simu))./(max(entropy_simu) - min(entropy_simu));
%     p_likelihood = 0.05; % set desired correlation
%     likelihood_simu = metafeature_simulation_creation(Prediction_accuracy,p_likelihood,Likelihood); % simulatelikelihood
%     likelihood_simu_norm = (likelihood_simu - mean(likelihood_simu))+1;

%%% let's try actual entropy instead of simulated
    %Entropy_norm = (Entropy - mean(Entropy))+1;
    %metafeature_array = [repmat(ones(n_subjects,5),1,1) entropy_simu_norm];

%%
% PLOT ACTUAL RELATIONSHIPS BETWEEN PREDICTIVE ACCURACY AND METAFEATURES
% To do:
% do this but for the Gaussianized version
% add the pearson and spearman correlation for each variable
% do for likelihood


% To do
% Then: mess about with p and method for obtaining stacking weights DONE
% Then: mess about with metafeatures simulation to see what works and what
% doesn't, e.g.
% - introduce a new metafeature 0.1 correlation doesn't do much for 1 metafeature, 
% but what about 2 metafeatures at 0.1 each? DONE
% - can we introduce a nonlinear relationship and still see improvements? Might need a nonlinear method

function meta_simu = metafeature_simulation_creation(pred_acc,p_meta,Metafeature)
    n_subjects = size(pred_acc,1);
    v_meta = randn(n_subjects, 1); % create random variable 'v'
    u_meta = (pred_acc ./ std(pred_acc)) - mean(pred_acc); % reverse: x = (s1 * u + m1)' to get u = randn(1, n);
    y_meta = std(Metafeature) .* squeeze((p_meta * u_meta + sqrt(1 - p_meta^2) * v_meta));
    y_meta = y_meta + mean(Metafeature) - mean(y_meta); % set mean of rv to desired mean
    meta_simu = y_meta;
end