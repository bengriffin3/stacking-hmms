function [pred_ST,pred_stack_ls,pred_stack_rdg,pred_FWLS_ls,pred_FWLS_rdg,W_stack_ls,W_stack_ridge,W_FWLS_ls,W_FWLS_ridge,pred_stack_RF,pred_FWLS_RF] =  ...
    predictPhenotype_mf_simulation_V2(pred_ST,vars_target,metafeature_array,non_nan_idx,folds_new)

% [mse_ST,mse_stack_ls,mse_stack_ridge,mse_FWLS_ls,mse_FWLS_ridge,ev_ST,ev_stack_ls,...
%     ev_stack_ridge,ev_FWLS_ls,ev_FWLS_ridge,prediction_stack_ls,prediction_stack_ridge,...
%     prediction_FWLS_ls,prediction_FWLS_ridge,w_stack_ls,w_stack_ridge,w_FWLS_ls,w_FWLS_ridge] = ...
%     predictPhenotype_mf_simulation_V2(Predictions,vars_target,metafeature_array)

%%% INPUTS
%%% Predictions: the predictions to be stacked
%%% vars_target: the variables to be predicted
%%% meatfeature_array: the metafeatures to be stacked

%%% OUTPUTS
%%% mse_ST: mean squared error for the original predictions
%%% mse_stack: mean squared error for the stacked prediction
%%% mse_FWLS: mean squared error for the stacked predictions w/ metafeatures
%%% ev_ST: explained variance for the  original predictions
%%% ev_stack: explained variance for the stacked prediction
%%% ev_FWLS: explained variance for the stacked predictions w/ metafeatures

n_subjects = size(vars_target,1);
n_repetitions = size(pred_ST,2);
n_metafeatures = (size(metafeature_array,2)/n_repetitions); % inc constant metafeature
n_folds = 10;

% Parameters for ridge regression reguarization paramater search
parameters = struct();
parameters.Method = 'ridge';
parameters.CVscheme = [10 10];
parameters.verbose = 0;

% set up cross validation folds
sub_partition = cvpartition(n_subjects,'KFold',10);



% initialise prediction variables and weights
pred_stack_ls = NaN(n_subjects,1);
pred_stack_rdg = NaN(n_subjects,1);
pred_stack_RF = NaN(n_subjects,1);
pred_FWLS_ls = NaN(n_subjects,1);
pred_FWLS_rdg = NaN(n_subjects,1);
pred_FWLS_RF = NaN(n_subjects,1);
W_stack_ls = NaN(n_repetitions*n_metafeatures,n_folds);
W_stack_ridge = NaN(n_repetitions*n_metafeatures,n_folds);
W_FWLS_ls = NaN(n_repetitions*n_metafeatures,n_folds);
W_FWLS_ridge = NaN(n_repetitions*n_metafeatures,n_folds);

for ifold = 1:n_folds
    
    % note test and training indices
    %test_idx = test(sub_partition,ifold);
    %training_idx = training(sub_partition,ifold);

    % note test and training indices USING SAME FOLDS AS USED FOR ORIGINAL PREDICTIONS
    a = folds_new{ifold}; b = zeros(n_subjects,1); b(a) = 1;
    test_idx = logical(b);
    training_idx = ~test_idx;

    % divide data into test and training sets
    predictions_simu_test = pred_ST(test_idx,:);
    predictions_simu_train = pred_ST(training_idx,:);
    %         vars_test = vars_target(test_idx);
    vars_train = vars_target(training_idx);
    
    % set up metafeature_prediction_array
    A = NaN(n_subjects,n_metafeatures*n_repetitions);
    for i = 1:n_metafeatures
        A(:,n_repetitions*i-n_repetitions+1:n_repetitions*i) = metafeature_array(:,n_repetitions*i-n_repetitions+1:n_repetitions*i).*pred_ST;
    end
    A_train = A(training_idx,:);
    A_test = A(test_idx,:);
    
    
    % This section is just checks I can remove later
    % corrcoef(Prediction_accuracy(:,1),A(:,6))
    % acc = (vars_target - Predictions).^2;
    % check correlation for first variable is still 0.4966
    % corrcoef(acc(:,1),A(:,6))
    
    % determine stacking weights
    opts1 = optimset('display','off');
    W_stack_ls(1:n_repetitions,ifold) = lsqlin(predictions_simu_train,vars_train,[],[],ones(1,n_repetitions),1,zeros(n_repetitions,1),ones(n_repetitions,1),[],opts1);
    %w_stack_ls = lsqlin(predictions_simu_train,vars_train,[],[],[],[],[],[],[],opts1);
    lambda = 0.0001;
    W_stack_ridge(1:n_repetitions,ifold) = (predictions_simu_train' * predictions_simu_train + lambda * eye(n_repetitions)) \ (predictions_simu_train' * vars_train);
   
    
    % determine metafeature stacking weights
    %[stats,~,~,~,~,~,lambda_best] = nets_predict5_BG(vars_train,A_train,'gaussian',parameters);
    lambda_best = 0.0001;
    lambda = lambda_best;
    
    % (i) Least Squares - constraints don't make sense once we add
    % metafeatures
    W_FWLS_ls(:,ifold) = lsqlin(A_train,vars_train,[],[],[],[],[],[],[],opts1); %w_FWLS_ls = lsqlin(A_train,vars_train,[],[],[ones(1,n_repetitions) zeros(1,n_repetitions*(n_metafeatures-1))],1,zeros(n_repetitions*n_metafeatures,1),ones(n_repetitions*n_metafeatures,1),[],opts1);
    % (ii) Ridge Regression
    W_FWLS_ridge(:,ifold) = (A_train' * A_train + lambda * eye(n_repetitions*n_metafeatures)) \ (A_train' * vars_train);
    
    % make stacked predictions
    pred_stack_ls(test_idx) = predictions_simu_test*W_stack_ls(1:n_repetitions,ifold);
    pred_stack_rdg(test_idx) = predictions_simu_test*W_stack_ridge(1:n_repetitions,ifold);
    
    % make stacked predictions with metafeatures
    pred_FWLS_ls(test_idx) = A_test*W_FWLS_ls(:,ifold);
    pred_FWLS_rdg(test_idx) = A_test*W_FWLS_ridge(:,ifold);

    % Make predictions using random forests
    % Stacked
%     Mdl_stack_rf = fitrensemble(predictions_simu_train,vars_train,'Method','Bag');
%     pred_stack_RF(test_idx) = predict(Mdl_stack_rf,predictions_simu_test);
% 
%     % Stacked with metafeatures
%     Mdl_FWLS_rf = fitrensemble(A_train,vars_train,'Method','Bag');
%     pred_FWLS_RF(test_idx) = predict(Mdl_FWLS_rf,A_test);


    
end



end

% sub_remove = sort(find(~non_nan_idx));
% if isempty(sub_remove)
%     folds_new = folds;
% else
%     folds_new = cell(n_folds,1);
%     for j = 1:n_folds
%         new_fold = folds{j};
%         % remove subjects without data points from folds
%         [X,Y] = ismember(sub_remove,new_fold);
%         new_fold(Y(X)) = [];
% 
%         % Reduce index of those larger than the removed subjects by 1
%         for i = 1:length(sub_remove)
%             idx_remove = sub_remove(i);
%             e = new_fold>idx_remove;
%             d = new_fold(new_fold>idx_remove) - 1;
%             new_fold(e) = d;
%             sub_remove = sub_remove - 1;
%         end
% 
% 
%         folds_new{j} = new_fold;
% 
%     end
% end



%     % note test and training indices
%     a = folds_new{ifold}; b = zeros(n_subjects,1); b(a) = 1;
%     test_idx = logical(b);
%     training_idx = ~test_idx;%setdiff(1:n_subjects,test_idx);
