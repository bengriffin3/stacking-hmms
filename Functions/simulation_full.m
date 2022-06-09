function [pred_ST,pred_stack_all,MSE_stack_ST,MSE_stack_all,EV_ST,EV_all,pearson_error_metafeature,Metafeature_simu,W_stack_all,squared_error] = simulation_full(vars,vars_hat,Metafeature,simulation_options)


% Note dimensions of arrays
n_subjects = size(vars_hat,1);
n_var = size(vars_hat,2);
n_repetitions = size(simulation_options.rep_selection,2);
n_metafeatures = (size(Metafeature,2)/n_repetitions) +1; % add 1 for constant metafeature

% Intialise variables
EV_ST = NaN(n_var,n_repetitions); EV_all = NaN(n_var,6,2);
MSE_stack_ST = NaN(n_var,n_repetitions); MSE_stack_all = NaN(n_var,6,2);
pred_ST = NaN(n_subjects,n_var,n_repetitions,2); pred_stack_all = NaN(1001,n_var,6,2);
W_stack_all = NaN(n_repetitions*n_metafeatures,simulation_options.n_folds,4,n_var,2);
Metafeature_simu = NaN(n_subjects,n_repetitions*(n_metafeatures-1),n_var);
pearson_error_metafeature = NaN(n_repetitions*(n_metafeatures-1),n_var);
squared_error = NaN(n_subjects,n_repetitions,n_var);

% Store selections
%vars_hat = vars_hat(:,:,simulation_options.rep_selection);
%Metafeature_all = Metafeature_all(:,simulation_options.rep_selection);
%Metafeature_gauss = Metafeature_gaussianized(:,simulation_options.rep_selection);


% remove (extreme) outliers
% we set outliers to NaN, so that then they are removed when we remove NaN subjects
% vars_outliers = NaN(n_subjects,n_var);
% for i = 1:34
%     [B,TF] = rmoutliers(vars(:,i),'percentile',[0.25 99.75]);
%     vars_outliers(~TF,i) = B;
% end
% vars = vars_outliers;


% % normalize variables and target (should we remove outliers before normalizing since this will change max/min?
% %  if we normalize each prediction, then we stack the predictions,
% % how do we de-normalize? since we change the predictions by different
% % amounts
vars_norm =  (vars-min(vars))./(max(vars)-min(vars));
vars_hat_norm = (vars_hat-min(vars_hat))./(max(vars_hat)-min(vars_hat));
Metafeature_norm = (Metafeature-min(Metafeature))./(max(Metafeature)-min(Metafeature)); % (Update: this does nothing - the weights just basically become very very small. If the metafeature values were super small, then this might matter because there is an upper limit on the weights)
vars_hat_norm = vars_hat_norm(:,:,simulation_options.rep_selection);

% Question: why does dividing by std work so well?
% vars_norm =  vars./nanstd(vars);
% vars_hat_norm = vars_hat./nanstd(vars_hat);
% Metafeature_norm = Metafeature_all./nanstd(Metafeature_all);

% Question: why does normalize() fct not work well?
% normalize() is actually standardization (mean 0 std 1)
%vars_norm = normalize(vars);
%vars_hat_norm = normalize(vars_hat);

% LOAD FOLDS AS USED FOR ORIGINAL PREDICTIONS
load('folds_all.mat','folds_all')


for v = 1:n_var
    v
    % remove NaNs
    [vars_target,Predictions,Metafeature,non_nan_idx] = nan_subject_remove(v,vars_norm,vars_hat_norm,Metafeature_norm);%,Squared_error_gaussianized); 

    % NOTE FOLDS USED FOR SPECIFIC VARIABLE FOR ORIGINAL PREDICTIONS
    folds_new = folds_all(:,v);

    % remove outliers
    %[vars_target,Predictions,Metafeature,squared_error_gaussianized] = outlier_remove(vars_target,Predictions,Metafeature,squared_error_gaussianized);

    % determine accuracy of predictions (we want our metafeature to be correlated with this)
    squared_error(non_nan_idx,:,v) = (vars_target - Predictions).^2;

    % simulate metafeature (to have same correlation as our actual metafeature)
    %Metafeature_simu(non_nan_idx,:,v) = metafeature_simulation_creation(squared_error,Metafeature,p_simulate,simulation_options);
    if isempty(simulation_options.simulated_metafeatures_in_r)
        [Metafeature_simu(non_nan_idx,:,v),pearson_error_metafeature(:,v)] = metafeature_simulation_creation(squared_error(non_nan_idx,:,v),Metafeature,simulation_options,vars_target);
    else
        Metafeature_simu(:,:,v) = simulation_options.simulated_metafeatures_in_r(:,:,v);
    end


    %     % set up metafeature array (note we add a 'constant' feature here
    %     if strcmp(simulation_options.metafeature_choice , 'Simulate')
    %         metafeature_array = [repmat(ones(size(Metafeature)),1,1) Metafeature_simu(non_nan_idx,:,var)];
    %     elseif strcmp(simulation_options.metafeature_choice , 'True')
    %         metafeature_array = [repmat(ones(size(Metafeature)),1,1) Metafeature];
    %     end

    % Make predictions and store prediction accuracies (simulated mf)
    metafeature_array_sim = [repmat(ones(size(Metafeature,1),n_repetitions),1,1) Metafeature_simu(non_nan_idx,:,v)];
    [pred_ST(non_nan_idx,v,:,1),pred_stack_ls,pred_stack_rdg,pred_FWLS_ls,pred_FWLS_rdg,W_stack_ls,W_stack_ridge,W_FWLS_ls,W_FWLS_ridge,pred_stack_RF,pred_FWLS_RF] = predictPhenotype_mf_simulation_V2(Predictions,vars_target,metafeature_array_sim,non_nan_idx,folds_new);
    pred_stack_all(non_nan_idx,v,:,1) = [pred_stack_ls pred_stack_rdg pred_FWLS_ls pred_FWLS_rdg pred_stack_RF,pred_FWLS_RF];
    W_stack_all(:,:,:,v,1) = cat(3,W_stack_ls,W_stack_ridge,W_FWLS_ls,W_FWLS_ridge);
    [MSE_stack_ST(v,:),MSE_stack_all(v,:,1),EV_ST(v,:),EV_all(v,:,1)] = prediction_accuracy_stats(vars_target,squeeze(pred_ST(non_nan_idx,v,:,1)),pred_stack_all(non_nan_idx,v,:,1));

    % Make predictions and store prediction accuracies (true mf)
    metafeature_array_real = [repmat(ones(size(Metafeature,1),n_repetitions),1,1) Metafeature];
    [pred_ST(non_nan_idx,v,:,2),pred_stack_ls,pred_stack_rdg,pred_FWLS_ls,pred_FWLS_rdg,W_stack_ls,W_stack_ridge,W_FWLS_ls,W_FWLS_ridge,pred_stack_RF,pred_FWLS_RF] = predictPhenotype_mf_simulation_V2(Predictions,vars_target,metafeature_array_real,non_nan_idx,folds_new);
    pred_stack_all(non_nan_idx,v,:,2) = [pred_stack_ls pred_stack_rdg pred_FWLS_ls pred_FWLS_rdg pred_stack_RF pred_FWLS_RF];
    W_stack_all(:,:,:,v,2) = cat(3,W_stack_ls,W_stack_ridge,W_FWLS_ls,W_FWLS_ridge);
    [~,MSE_stack_all(v,:,2),~,EV_all(v,:,2)] = prediction_accuracy_stats(vars_target,squeeze(pred_ST(non_nan_idx,v,:,2)),pred_stack_all(non_nan_idx,v,:,2));


end
squared_error = repmat(squared_error,1,n_metafeatures);
end