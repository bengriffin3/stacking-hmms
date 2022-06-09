function [pa_ST,pa_stack,pa_FWLS,ev_ST,ev_stack,ev_FWLS] = ...
    predictPhenotype_mf_simulation(Predictions,vars_target,metafeature_array)
%%% INPUTS
%%% Predictions: the predictions to be stacked
%%% vars_target: the variables to be predicted
%%% meatfeature_array: the metafeatures to be stacked

%%% OUTPUTS
%%% pa_ST: mean squared error for the original predictions
%%% pa_stack: mean squared error for the stacked prediction
%%% pa_FWLS: mean squared error for the stacked predictions w/ metafeatures
%%% ev_ST: explained variance for the  original predictions
%%% ev_stack: explained variance for the stacked prediction
%%% ev_FWLS: explained variance for the stacked predictions w/ metafeatures

n_subjects = size(vars_target,1);
n_repetitions = size(Predictions,2);
n_metafeatures = size(metafeature_array,2)/n_repetitions;

% Parameters for ridge regression reguarization paramater search
parameters = struct();
parameters.Method = 'ridge';
parameters.CVscheme = [10 10];
parameters.verbose = 0;



% set up cross validation folds
sub_partition = cvpartition(n_subjects,'KFold',10);
% idxTrain = training(sub_partition);

% initialise prediction + explained variance variables
prediction_stack = NaN(n_subjects,1);
prediction_FWLS = NaN(n_subjects,1);

for ifold = 1:10
    
    % note test and training indices
    test_idx = test(sub_partition,ifold);
    training_idx = training(sub_partition,ifold);
    
    % divide data into test and training sets
    predictions_simu_test = Predictions(test_idx,:);
    predictions_simu_train = Predictions(training_idx,:);
    %         vars_test = vars_target(test_idx);
    vars_train = vars_target(training_idx);
    
    % set up metafeature_prediction_array
    A = NaN(n_subjects,n_metafeatures*n_repetitions);
    for i = 1:n_metafeatures
        A(:,n_repetitions*i-n_repetitions+1:n_repetitions*i) = metafeature_array(:,n_repetitions*i-n_repetitions+1:n_repetitions*i).*Predictions;
    end
    A_train = A(training_idx,:);
    A_test = A(test_idx,:);
    
    % This section is just checks I can remove later
    %corrcoef(Prediction_accuracy(:,1),A(:,6))
    % acc = (vars_target - Predictions).^2;
    % check correlation for first variable is still 0.4966
    % corrcoef(acc(:,var,1),A(:,6,var))
    
    % determine stacking weights
    opts1 = optimset('display','off');
    %w_stack = lsqlin(squeeze(predictions_simu_train(:,var,:)),vars_train(:,var),[],[],ones(1,n_repetitions),1,zeros(n_repetitions,1),ones(n_repetitions,1),[],opts1);
    w_stack = lsqlin(predictions_simu_train,vars_train,[],[],ones(1,n_repetitions),1,zeros(n_repetitions,1),ones(n_repetitions,1),[],opts1);
    
    % I think I need to change this to get it working (currenty they want
    % to set metafeature coefficients to 0
    % also need to add in mutiple metafeatures, because currently a
    % metafeature needs p = 0.5+ correlation to make a difference, but what
    % if have multiple features each with p = 0.3 or something
    
    % determine metafeature stacking weights
    [stats,~,~,~,~,~,lambda_best] = nets_predict5_BG(vars_train,A_train,'gaussian',parameters);
    lambda = lambda_best;
    
    A_train_v = A_train;
    % (i) Least Squares
    w_FWLS = lsqlin(A_train,vars_train,[],[],[ones(1,n_repetitions) zeros(1,n_repetitions*(n_metafeatures-1))],1,zeros(n_repetitions*n_metafeatures,1),ones(n_repetitions*n_metafeatures,1),[],opts1);
    % (ii) Ridge Regression
    %w_FWLS = (A_train_v' * A_train_v + lambda * eye(n_repetitions*n_metafeatures)) \ (A_train_v' * vars_train);
    
    % make stacked predictions
    prediction_stack(test_idx) = predictions_simu_test*w_stack;
    prediction_FWLS(test_idx) = A_test*w_FWLS;
    
end


% Prediction accuracies
pa_ST = sum((vars_target - Predictions).^2)/n_subjects;
pa_stack = sum((vars_target - prediction_stack).^2)/n_subjects;
pa_FWLS = sum((vars_target - prediction_FWLS).^2)/n_subjects;

% Explained variance  
ev_ST = corr(Predictions,vars_target).^2;
ev_stack = corr(prediction_stack,vars_target).^2;
ev_FWLS = corr(prediction_FWLS,vars_target).^2;

end

