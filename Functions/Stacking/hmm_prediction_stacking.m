function [vars_hat_stack_ls, vars_hat_stack_rdg, vars_hat_stack_RF, W_stack_ls, W_stack_rdg] = hmm_prediction_stacking(y,pred_ST,folds)

%%% INPUTS
%%% pred_ST: the predictions to be stacked
%%% y: the variables to be predicted
%%% folds: number of folds used for original HMM predictions, which will also be used here

%%% OUTPUTS
%%% vars_hat_stack: stacked prediction
%%% explained_variance_stack: explained variance for the stacked predicti


vars_target = y;

n_subjects = length(y);
n_repetitions = size(pred_ST,2);
n_folds = length(folds);

% initialise prediction variables and weights
vars_hat_stack_ls = NaN(n_subjects,1);
vars_hat_stack_rdg = NaN(n_subjects,1);
vars_hat_stack_RF = NaN(n_subjects,1);
W_stack_ls = NaN(n_repetitions,n_folds);
W_stack_rdg = NaN(n_repetitions,n_folds);
%explained_variance_stack_ls = NaN(1);
%explained_variance_stack_ridge = NaN(1);


for ifold = 1:n_folds

    % note test and training indices USING SAME FOLDS AS USED FOR ORIGINAL PREDICTIONS
    a = folds{ifold}; b = zeros(n_subjects,1); b(a) = 1;
    test_idx = logical(b);
    training_idx = ~test_idx;

    % divide data into test and training sets
    predictions_test = pred_ST(test_idx,:);
    predictions_train = pred_ST(training_idx,:);
    %         vars_test = vars_target(test_idx);
    vars_train = vars_target(training_idx);


    % determine stacking weights
    opts1 = optimset('display','off');
    %W_stack_ls(1:n_repetitions,ifold) = lsqlin(predictions_train,vars_train,[],[],ones(1,n_repetitions),1,zeros(n_repetitions,1),ones(n_repetitions,1),[],opts1);
    W_stack_ls(1:n_repetitions,ifold) = lsqlin(predictions_train,vars_train,[],[],ones(1,n_repetitions),1,zeros(n_repetitions,1),ones(n_repetitions,1),[],opts1);
    %w_stack_ls = lsqlin(predictions_simu_train,vars_train,[],[],[],[],[],[],[],opts1);
    lambda = 0.0001;
    W_stack_rdg(1:n_repetitions,ifold) = (predictions_train' * predictions_train + lambda * eye(n_repetitions)) \ (predictions_train' * vars_train);

    % make stacked predictions
    vars_hat_stack_ls(test_idx) = predictions_test*W_stack_ls(1:n_repetitions,ifold);
    vars_hat_stack_rdg(test_idx) = predictions_test*W_stack_rdg(1:n_repetitions,ifold);

%     % Make stacked predictions using random forests
%     Mdl_stack_rf = fitrensemble(predictions_train,vars_train,'Method','Bag');
%     vars_hat_stack_RF(test_idx) = predict(Mdl_stack_rf,predictions_test);

end



end

%     % determine metafeature stacking weights
%     %[stats,~,~,~,~,~,lambda_best] = nets_predict5_BG(vars_train,A_train,'gaussian',parameters);
%     lambda_best = 0.0001;
%     lambda = lambda_best;
